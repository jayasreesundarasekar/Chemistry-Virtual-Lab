"""ChemLab API – Flask + RDKit. Run locally: python app.py  (http://localhost:5000)"""
from flask import Flask, request, jsonify
from flask_cors import CORS
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, Lipinski, rdMolDescriptors, Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog("rdApp.*")
app = Flask(__name__)
CORS(app)


def embed_3d(mol):
    """ETKDG embedding + MMFF/UFF optimisation. Returns an SDF/molblock string or None."""
    mh = Chem.AddHs(mol)
    if mh.GetNumAtoms() > 300:
        return None
    ok = AllChem.EmbedMolecule(mh, randomSeed=42)
    if ok != 0:
        ok = AllChem.EmbedMolecule(mh, useRandomCoords=True, randomSeed=7)
    if ok != 0:
        return None
    try:
        if AllChem.MMFFHasAllMoleculeParams(mh):
            AllChem.MMFFOptimizeMolecule(mh, maxIters=500)
        else:
            AllChem.UFFOptimizeMolecule(mh, maxIters=500)
    except Exception:
        pass
    return Chem.MolToMolBlock(mh)


def analyze(smiles, embed=True):
    smiles = (smiles or "").strip()
    if not smiles or len(smiles) > 500:
        return {"ok": False}
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return {"ok": False}

    mh = Chem.AddHs(mol)
    kek = Chem.Mol(mh)
    Chem.Kekulize(kek, clearAromaticFlags=True)
    sigma = pi = 0
    for b in kek.GetBonds():
        sigma += 1
        pi += {Chem.BondType.DOUBLE: 1, Chem.BondType.TRIPLE: 2}.get(b.GetBondType(), 0)
    heavy_bonds = mol.GetNumBonds()

    d = rdMolDraw2D.MolDraw2DSVG(300, 220)
    d.DrawMolecule(mol)
    d.FinishDrawing()

    return {
        "ok": True,
        "canon": Chem.MolToSmiles(mol),
        "svg": d.GetDrawingText(),
        "formula": rdMolDescriptors.CalcMolFormula(mol),
        "hv": mol.GetNumHeavyAtoms(),
        "natoms": mh.GetNumAtoms(),
        "heavyBonds": heavy_bonds,
        "sig": sigma,
        "pi": pi,
        "P": {
            "mw": Descriptors.MolWt(mol),
            "exact": Descriptors.ExactMolWt(mol),
            "logp": Crippen.MolLogP(mol),
            "tpsa": rdMolDescriptors.CalcTPSA(mol),
            "hbd": Lipinski.NumHDonors(mol),
            "hba": Lipinski.NumHAcceptors(mol),
            "rot": rdMolDescriptors.CalcNumRotatableBonds(mol),
            "rings": rdMolDescriptors.CalcNumRings(mol),
            "fsp3": rdMolDescriptors.CalcFractionCSP3(mol),
        },
        "sdf": embed_3d(mol) if embed else None,
    }


@app.get("/")
def health():
    return jsonify(status="ChemLab API running")


@app.post("/api/analyze")
def api_analyze():
    data = request.get_json(silent=True) or {}
    return jsonify(analyze(data.get("smiles"), data.get("embed", True)))


if __name__ == "__main__":
    app.run(debug=True, port=5000)
