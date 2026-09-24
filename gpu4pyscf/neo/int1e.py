import math
import ctypes
import numpy as np
import cupy as cp
from pyscf import gto
from pyscf.gto.mole import ATOM_OF

from gpu4pyscf.gto.mole import (
    SortedGTO, PBCIntEnvVars, extract_pgto_params, _scale_sp_ctr_coeff)
from gpu4pyscf.lib.cupy_helper import asarray, hermi_triu
from gpu4pyscf.pbc.df.ft_ao import libpbc
from gpu4pyscf.pbc.gto.int1e import L_AUX_MAX
from gpu4pyscf.pbc.gto import int1e


libpbc.PBCint1e_ovlp_multi_out.restype = ctypes.c_int
libpbc.PBCint1e_kin_multi_out.restype = ctypes.c_int
libpbc.PBCint1e_ipovlp_multi_out.restype = ctypes.c_int
libpbc.PBCint1e_ipkin_multi_out.restype = ctypes.c_int


def _one_center_basis_key(mol):
    # One-center overlap and kinetic integrals are translation invariant.
    if len(np.unique(mol._bas[:,ATOM_OF])) != 1:
        return None
    shells = []
    for ib in range(mol.nbas):
        coeff = mol.bas_ctr_coeff(ib)
        shells.append((mol.bas_angular(ib), mol.bas_kappa(ib),
                       tuple(mol.bas_exp(ib)), coeff.shape,
                       tuple(coeff.ravel())))
    return mol.cart, tuple(shells)


def _shell_overlap_mask(mol, bas_ij_idx, precision=1e-14):
    # The molecular branch is copied from pbc.gto.int1e._shell_overlap_mask.
    exps, cs = extract_pgto_params(mol, 'diffuse')
    exps = cp.asarray(exps, dtype=np.float32)
    log_coeff = cp.log(abs(asarray(cs, dtype=np.float32)))
    ao_loc = cp.zeros(1, dtype=np.int32)
    Ls = cp.zeros((1, 3))
    nimgs = len(Ls)
    # NEO: one screening flag per candidate, not per concatenated shell pair.
    ovlp_mask = cp.zeros(len(bas_ij_idx), dtype=bool)
    envs = PBCIntEnvVars.new(
        mol.natm, mol.nbas, nimgs, nimgs, asarray(mol._atm),
        asarray(mol._bas), asarray(_scale_sp_ctr_coeff(mol)), ao_loc, Ls)
    libpbc.PBCovlp_mask_estimation_indexed(
        ctypes.cast(ovlp_mask.data.ptr, ctypes.c_void_p),
        ctypes.cast(exps.data.ptr, ctypes.c_void_p),
        ctypes.cast(log_coeff.data.ptr, ctypes.c_void_p),
        ctypes.byref(envs), ctypes.c_float(math.log(precision)),
        ctypes.cast(bas_ij_idx.data.ptr, ctypes.c_void_p),
        ctypes.c_size_t(len(bas_ij_idx)))
    # End copied block.
    return ovlp_mask


def _generate_shl_pairs(mol, shell_component, hermi=1, precision=1e-14, tile=1):
    # SortedMole.generate_shl_pairs uses ish*nbas+jsh within angular blocks.
    # NEO: construct only component-local candidates before GPU screening.
    nbas = mol.nbas
    bas_ij_idx = []
    for ic in np.unique(shell_component):
        idx = np.where(shell_component == ic)[0]
        if hermi:
            i, j = np.tril_indices(len(idx))
            bas_ij_idx.append(idx[i] * nbas + idx[j])
        else:
            bas_ij_idx.append((idx[:,None] * nbas + idx).ravel())
    bas_ij_idx = np.sort(np.hstack(bas_ij_idx))
    ish, jsh = divmod(bas_ij_idx, nbas)
    l_ctr_offsets = np.append(0, np.cumsum(mol.l_ctr_counts))
    groups = len(mol.uniq_l_ctr)
    if hermi == 1:
        ij_tasks = [(i, j) for i in range(groups) for j in range(i+1)]
    else:
        ij_tasks = [(i, j) for i in range(groups) for j in range(groups)]
    bas_ij_cache = {}
    for i, j in ij_tasks:
        ish0, ish1 = l_ctr_offsets[i], l_ctr_offsets[i+1]
        jsh0, jsh1 = l_ctr_offsets[j], l_ctr_offsets[j+1]
        mask = (ish >= ish0) & (ish < ish1) & (jsh >= jsh0) & (jsh < jsh1)
        pair_ij = bas_ij_idx[mask]
        if tile > 1:
            # Preserve PBCsort_pair_ij's tile-i, tile-j, i, j traversal,
            # including partial tiles, without enumerating cross-component pairs.
            irel = ish[mask] - ish0
            jrel = jsh[mask] - jsh0
            order = np.lexsort((jrel % tile, irel % tile, jrel // tile, irel // tile))
            pair_ij = pair_ij[order]
        bas_ij_cache[i,j] = pair_ij
    counts = [len(pair) for pair in bas_ij_cache.values()]
    bas_ij_idx = cp.asarray(np.hstack(list(bas_ij_cache.values())), dtype=np.int32)
    # Screen all angular blocks in one launch and retain compact storage.
    mask = _shell_overlap_mask(mol, bas_ij_idx, precision)
    offsets = np.cumsum(counts[:-1])
    pairs = cp.split(bas_ij_idx, offsets)
    masks = cp.split(mask, offsets)
    return {key: pair[keep] for key, pair, keep in zip(bas_ij_cache, pairs, masks)}


class _Int1eOpt(int1e._Int1eOpt):
    def __init__(self, cell, atom_component, hermi=0):
        # Copied from pbc.gto.int1e._Int1eOpt.__init__ for Mole inputs.
        # NEO: atom_component identifies the owner of each atom before sorting.
        self.cell = cell = SortedGTO.from_cell(cell, decontract=True)
        lmax = self.cell.uniq_l_ctr[:,0].max()
        assert lmax <= L_AUX_MAX

        bvk_ncells = 1
        bvk_kmesh = None
        bvkcell = cell
        bvkmesh_Ls = Ls = cp.zeros((1, 3))
        self.hermi = hermi
        self.bvk_kmesh = bvk_kmesh
        self.bvkcell = bvkcell
        self.bvkmesh_Ls = bvkmesh_Ls

        _env = _scale_sp_ctr_coeff(bvkcell)
        self.int1e_envs = PBCIntEnvVars.new(
            cell.natm, cell.nbas, bvk_ncells, len(Ls),
            bvkcell._atm, bvkcell._bas, _env, cell.p_ao_loc, Ls)

        # NEO: apply component locality before the original shell-pair
        # aggregation instead of screening the concatenated square pair list.
        shell_component = atom_component[cell._bas[:,ATOM_OF]]
        bas_ij_cache = _generate_shl_pairs(cell, shell_component, hermi)
        bas_ij_idx, shl_pair_offsets = cell.aggregate_shl_pairs(bas_ij_cache)
        self.bas_ij_cache = bas_ij_cache
        self.bas_ij_idx = bas_ij_idx
        self.shl_pair_offsets = shl_pair_offsets
        # End copied block.


class Int1eOpt:
    def __init__(self, components, hermi=1):
        self.components = components
        self.component_names = list(components)
        # Evaluate translated copies of the same one-center basis only once.
        representatives = {}
        component_representative = {}
        for key, mol in components.items():
            basis_key = _one_center_basis_key(mol)
            if basis_key is None:
                representative = key
            else:
                representative = representatives.setdefault(basis_key, key)
            component_representative[key] = representative
        integral_names = list(dict.fromkeys(component_representative.values()))
        integral_components = {key: components[key] for key in integral_names}
        mols = list(integral_components.values())
        # Concatenation creates one integral work list.  It does not introduce
        # cross-component AO blocks in the outputs below.
        mol = mols[0]
        atom_component = [0] * mol.natm
        for ic, mol_t in enumerate(mols[1:], 1):
            mol = gto.conc_mol(mol, mol_t)
            atom_component.extend([ic] * mol_t.natm)

        mol = SortedGTO.from_cell(mol, decontract=True)
        # Sorting changes shell indices, so component ownership is assigned in
        # the same sorted shell order used by the original optimizer.
        shell_component = np.asarray(atom_component, dtype=np.int32)[mol._bas[:,ATOM_OF]]
        self._opt = opt = _Int1eOpt(mol, np.asarray(atom_component, dtype=np.int32), hermi=hermi)
        self.mol = mol = opt.cell
        nbas = mol.nbas
        # These are the original angular-momentum work blocks after removing
        # shell pairs whose two AOs belong to different components.
        self.bas_ij_idx = opt.bas_ij_idx
        self.shl_pair_offsets = opt.shl_pair_offsets

        component_mols = {
            key: SortedGTO.from_cell(component, decontract=True)
            for key, component in integral_components.items()}
        shell_local = np.concatenate(
            [np.arange(component.nbas) for component in component_mols.values()]
        )[mol.sorted_idx]
        # The integral environment uses concatenated shell indices, whereas
        # every output uses its component-local AO indices.
        local_ao_loc = np.empty(nbas, dtype=np.int32)
        component_nao = np.empty(len(integral_names), dtype=np.int32)
        for ic, key in enumerate(integral_names):
            component = component_mols[key]
            component_nao[ic] = component.nao
            inv_sorted = np.empty_like(component.sorted_idx)
            inv_sorted[component.sorted_idx] = np.arange(component.nbas)
            idx = np.where(shell_component == ic)[0]
            local_ao_loc[idx] = component.ao_loc[inv_sorted[shell_local[idx]]]

        self.component_mols = component_mols
        self.integral_names = integral_names
        self.component_representative = component_representative
        self.shell_component = cp.asarray(shell_component, dtype=np.int32)
        self.local_ao_loc = cp.asarray(local_ao_loc, dtype=np.int32)
        self.component_nao = cp.asarray(component_nao, dtype=np.int32)

    def intor(self, name, comp, deriv, hermi=0):
        gout_stride, shm_size = int1e._gout_stride_lookup_table(self.mol, deriv)
        # Allocate one dense matrix per component instead of a zero-padded
        # matrix for the concatenated AO space.
        out = {}
        for key, component in self.component_mols.items():
            shape = (component.nao, component.nao)
            if comp > 1:
                shape = (comp,) + shape
            out[key] = cp.zeros(shape)
        out_ptrs = cp.asarray(np.asarray(
            [out[key].data.ptr for key in self.integral_names],
            dtype=np.uintp))
        # The multi-output driver follows the original integral kernel but
        # selects the destination pointer and local AO offsets by component.
        drv = getattr(libpbc, name)
        err = drv(
            ctypes.cast(out_ptrs.data.ptr, ctypes.c_void_p),
            ctypes.byref(self._opt.int1e_envs), ctypes.c_int(shm_size),
            ctypes.c_int(len(self.shl_pair_offsets) - 1),
            ctypes.cast(self.bas_ij_idx.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shl_pair_offsets.data.ptr, ctypes.c_void_p),
            ctypes.cast(gout_stride.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.shell_component.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.local_ao_loc.data.ptr, ctypes.c_void_p),
            ctypes.cast(self.component_nao.data.ptr, ctypes.c_void_p))
        if err != 0:
            raise RuntimeError(f'{name} failed')
        for key in self.integral_names:
            if hermi:
                out[key] = hermi_triu(out[key], hermi=hermi, inplace=True)
            out[key] = self.component_mols[key].apply_CT_mat_C(out[key])
        return {key: out[self.component_representative[key]] for key in self.component_names}

    def get_ovlp(self):
        return self.intor('PBCint1e_ovlp_multi_out', 1, (0, 0), hermi=1)

    def get_kin(self):
        return self.intor('PBCint1e_kin_multi_out', 1, (2, 0), hermi=1)

    def get_ipovlp(self):
        return self.intor('PBCint1e_ipovlp_multi_out', 3, (1, 0))

    def get_ipkin(self):
        return self.intor('PBCint1e_ipkin_multi_out', 3, (3, 0))
