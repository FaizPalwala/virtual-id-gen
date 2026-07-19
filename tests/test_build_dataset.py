import numpy as np
import pandas as pd
from builddataset import OUTPUT_COLUMNS, build_dataset

def test_build_preserves_legacy_csv_schema(tmp_path):
    identities, embeddings, output = tmp_path / "identities", tmp_path / "emb", tmp_path / "out"
    identities.mkdir(); embeddings.mkdir()
    rows = [{"imagepath": f"image_{cluster}_{number}.jpg", "clusterid": cluster} for cluster in range(3) for number in range(2)]
    pd.DataFrame(rows).to_csv(identities / "identitymanifest.csv", index=False)
    np.save(embeddings / "imagepaths.npy", np.array([row["imagepath"] for row in rows]))
    np.save(embeddings / "agegroups.npy", np.zeros(len(rows), dtype=np.int8))
    result = pd.read_csv(build_dataset(str(identities), str(embeddings), str(output), nforget=1, ntest=1))
    assert list(result.columns) == OUTPUT_COLUMNS
    assert len(result) == 6
