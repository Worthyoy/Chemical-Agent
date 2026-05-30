import pubchempy as pcp


SMILES = "OB(O)c1ccc(F)cc1"


def smiles_to_iupac(smiles: str) -> str | None:
    compounds = pcp.get_compounds(smiles, "smiles")
    if not compounds:
        return None
    return compounds[0].iupac_name


if __name__ == "__main__":
    iupac_name = smiles_to_iupac(SMILES)
    print(iupac_name if iupac_name else "IUPAC name not found")
