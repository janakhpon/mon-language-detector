import sys

import pandas as pd

from mon_language_detector.wrangle import CorpusCleaner, main


def test_tsv_transformation(tmp_path):
    src = tmp_path / "test.tsv"
    dst = tmp_path / "test_cleaned.txt"

    df = pd.DataFrame({"id": [1, 2], "text": ["Hello \u200bWorld", "Test line"]})
    df.to_csv(src, sep="\t", index=False)

    CorpusCleaner().process_file(src, dst, text_col="text", header=True)

    assert dst.exists()
    lines = dst.read_text(encoding="utf-8").splitlines()
    assert lines == ["Hello World", "Test line"]


def test_batch_processing(tmp_path):
    src_dir = tmp_path / "raw"
    dst_dir = tmp_path / "clean"
    src_dir.mkdir()

    (src_dir / "f1.tsv").write_text("id\ttext\n1\tLine One\n2\tLine Two", encoding="utf-8")
    (src_dir / "f2.csv").write_text("id,text\n1,Line Three\n2,Line Four", encoding="utf-8")

    sys.argv = ["wrangle", "--input", str(src_dir), "--output", str(dst_dir), "--header", "true"]
    main()

    assert (dst_dir / "f1_cleaned.txt").exists()
    assert (dst_dir / "f2_cleaned.txt").exists()

    lines_f1 = (dst_dir / "f1_cleaned.txt").read_text(encoding="utf-8").splitlines()
    assert lines_f1 == ["Line One", "Line Two"]

    lines_f2 = (dst_dir / "f2_cleaned.txt").read_text(encoding="utf-8").splitlines()
    assert lines_f2 == ["Line Three", "Line Four"]
