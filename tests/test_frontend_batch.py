"""CSV batch tests. HTTP is MOCKED (httpx.MockTransport) except where a test says otherwise; mocked responses are crafted
fixtures and do not show that a real trained model works."""
import io
import json

import httpx
import pandas as pd
import pytest
from frontend import batch
from frontend.api_client import ApiClient
import fixture_builders as fb

HDR = "pair_id,original_text,simplified_text,language,context,legal_domain\n"


def csv_bytes(rows, header=HDR):
    return (header + "".join(rows)).encode("utf-8")


def recording_client(available=False, errors=None, fail_on_call=None, status_on_call=None, record=None):
    """Mock backend that records each /batch-predict request and answers with crafted results."""
    state = {"n": 0}

    def handler(request):
        state["n"] += 1
        body = json.loads(request.content)
        if record is not None:
            record.append(body)
        if fail_on_call == state["n"]:
            raise httpx.ConnectError("refused")
        if status_on_call and state["n"] in status_on_call:
            return httpx.Response(status_on_call[state["n"]], json={"error": {"code": "validation_error", "message": "bad batch"}})
        resp = fb.batch_response(body["items"], available=available, errors=(errors or {}).get(state["n"]))
        return httpx.Response(200, content=json.dumps(resp), headers={"content-type": "application/json"})
    return ApiClient("http://backend.test", transport=httpx.MockTransport(handler))


# ------------------------------------------------------------------ file-level validation
def test_valid_file():
    p = batch.parse_csv(csv_bytes(["a1,The tenant must pay.,The tenant may pay.,en,ctx,rental\n", "a2,Pay Rs. 5.,Pay Rs. 6.,,,\n"]))
    assert p.file_errors == [] and p.row_errors == [] and (p.n_valid, p.n_invalid) == (2, 0)
    assert p.table["language_used"].tolist() == ["en", "en"]  # blank language defaults to English


@pytest.mark.parametrize("data,needle", [
    (b"", "empty"),
    (b"original_text,simplified_text\n", "no data rows"),
    (b"original_text,language\nA,en\n", "simplified_text"),
    (b"simplified_text,language\nA,en\n", "original_text"),
    (b"a,b\n1,2\n", "original_text, simplified_text"),
    (b"\xff\xfe\x00bad", "UTF-8"),
])
def test_file_errors(data, needle):
    p = batch.parse_csv(data)
    assert p.table is None and len(p.file_errors) == 1 and needle in p.file_errors[0]


def test_missing_columns_message_lists_what_was_found():
    p = batch.parse_csv(b"orig,simp\nA,B\n")
    assert "Missing required column(s): original_text, simplified_text" in p.file_errors[0] and "orig, simp" in p.file_errors[0]


def test_malformed_csv_is_reported_not_raised():
    p = batch.parse_csv(b'original_text,simplified_text\n"unterminated,B\nC,D,E,F,G\n')
    assert p.table is not None or p.file_errors  # either parsed leniently or reported; never an exception


def test_size_and_row_limits(monkeypatch):
    monkeypatch.setattr(batch, "MAX_UPLOAD_BYTES", 50)
    assert "limit" in batch.parse_csv(b"original_text,simplified_text\n" + b"x,y\n" * 50).file_errors[0]
    monkeypatch.undo()
    monkeypatch.setattr(batch, "MAX_ROWS", 3)
    p = batch.parse_csv(b"original_text,simplified_text\n" + b"x,y\n" * 4)
    assert p.table is None and "limit is 3" in p.file_errors[0]  # never silently truncated


def test_bom_and_header_normalisation():
    data = "\ufeff Original_Text ,SIMPLIFIED_TEXT\nA,B\n".encode("utf-8")
    p = batch.parse_csv(data)
    assert p.file_errors == [] and p.table.loc[0, "original_text"] == "A"


def test_extra_columns_ignored_and_na_text_kept_literally():
    p = batch.parse_csv(b"original_text,simplified_text,notes\nNA,null,whatever\n")
    assert p.table.loc[0, "original_text"] == "NA" and p.table.loc[0, "simplified_text"] == "null"


# ------------------------------------------------------------------ row-level validation
def test_invalid_rows_get_row_specific_errors_and_are_kept():
    rows = ["r1,Good original.,Good simplified.,en,,\n", "r2,,Missing original.,en,,\n", "r3,Fine.,   ,en,,\n",
            "r4,Hindi row.,Hindi simple.,hi,,\n", "r5,Odd.,Odd.,e!,,\n", f"r6,{'x' * 5001},ok,en,,\n",
            f"r7,ok,ok,en,,{'d' * 101}\n", "r8,Good.,Good.,EN-in,,\n"]
    p = batch.parse_csv(csv_bytes(rows))
    assert len(p.table) == 8 and p.n_valid == 2 and p.n_invalid == 6
    by_row = {}
    for e in p.row_errors:
        by_row.setdefault(e.row_number, []).append((e.field, e.input_id))
    assert by_row == {2: [("original_text", "r2")], 3: [("simplified_text", "r3")], 4: [("language", "r4")],
                      5: [("language", "r5")], 6: [("original_text", "r6")], 7: [("legal_domain", "r7")]}
    assert "not supported" in [e.message for e in p.row_errors if e.row_number == 4][0]
    assert p.table["valid"].tolist() == [True, False, False, False, False, False, False, True]


def test_whitespace_only_required_fields_are_invalid():
    p = batch.parse_csv(b"original_text,simplified_text\n \t ,B\nA,\n")
    assert p.n_invalid == 2


def test_supported_languages_come_from_the_backend_not_hardcoded():
    rows = ["r1,A.,B.,hi,,\n"]
    assert batch.parse_csv(csv_bytes(rows), ["en"]).n_invalid == 1
    assert batch.parse_csv(csv_bytes(rows), ["en", "hi"]).n_valid == 1


def test_row_ids_and_numbers_are_preserved():
    p = batch.parse_csv(csv_bytes(["x-9,A.,B.,en,,\n", ",C.,D.,en,,\n", "x-9,A.,B.,en,,\n"]))  # duplicates / blank ids kept
    assert p.table["row_number"].tolist() == [1, 2, 3] and p.table["input_id"].tolist() == ["x-9", "", "x-9"]
    for col, header in (("id", "id,original_text,simplified_text\n"), ("row_id", "row_id,original_text,simplified_text\n")):
        q = batch.parse_csv((header + "k1,A,B\n").encode())
        assert q.table.loc[0, "input_id"] == "k1", col
    assert batch.parse_csv(b"original_text,simplified_text\nA,B\n").table.loc[0, "input_id"] == ""


def test_build_items_only_valid_rows_and_optional_fields_only_when_present():
    p = batch.parse_csv(csv_bytes(["a,A.,B.,en,some ctx,rental\n", "b,,B.,en,,\n", "c,C.,D.,,,\n"]))
    nums, items = batch.build_items(p.table)
    assert nums == [1, 3]
    assert items[0] == {"original_text": "A.", "simplified_text": "B.", "language": "en", "context": "some ctx", "legal_domain": "rental"}
    assert items[1] == {"original_text": "C.", "simplified_text": "D.", "language": "en"}


# ------------------------------------------------------------------ batching (>100 records)
def many_rows(n):
    return csv_bytes([f"id{i},Sentence {i} must pay.,Sentence {i} may pay.,en,,\n" for i in range(n)])


@pytest.mark.parametrize("n,sizes", [(1, [1]), (100, [100]), (101, [100, 1]), (250, [100, 100, 50]), (300, [100, 100, 100])])
def test_requests_never_exceed_100_items(n, sizes):
    seen = []
    p = batch.parse_csv(many_rows(n))
    out = batch.run_batch(recording_client(record=seen), p.table)
    assert [len(b["items"]) for b in seen] == sizes and max(sizes) <= 100
    assert len(out) == n and all(s == "ok" for s, _ in out.values())


def test_chunk_helper_never_exceeds_the_api_limit():
    assert [len(c) for c in batch.chunked(list(range(250)), size=500)] == [100, 100, 50]
    assert list(batch.chunked([])) == []


def test_results_stay_aligned_with_rows_when_invalid_rows_are_interleaved():
    rows = [f"id{i},{'' if i % 3 == 1 else 'Orig ' + str(i) + '.'},Simp {i}.,en,,\n" for i in range(7)]
    p = batch.parse_csv(csv_bytes(rows))
    seen = []
    out = batch.run_batch(recording_client(record=seen), p.table)
    df = batch.assemble(p.table, out)
    assert df["row_number"].tolist() == list(range(1, 8)) and df["input_id"].tolist() == [f"id{i}" for i in range(7)]
    assert df["status"].tolist() == ["ok", "invalid_input", "ok", "ok", "invalid_input", "ok", "ok"]
    sent = [it["original_text"] for b in seen for it in b["items"]]
    assert sent == ["Orig 0.", "Orig 2.", "Orig 3.", "Orig 5.", "Orig 6."]  # invalid rows were never sent
    assert df.loc[df["status"] == "invalid_input", "error_message"].str.contains("original_text").all()


def test_progress_callback_reports_each_chunk():
    calls = []
    batch.run_batch(recording_client(), batch.parse_csv(many_rows(205)).table, progress=lambda d, t: calls.append((d, t)))
    assert calls == [(1, 3), (2, 3), (3, 3)]


# ------------------------------------------------------------------ failures while sending
def test_backend_offline_midway_marks_remaining_rows_not_sent():
    p = batch.parse_csv(many_rows(250))
    out = batch.run_batch(recording_client(fail_on_call=2), p.table)
    df = batch.assemble(p.table, out)
    counts = df["status"].value_counts().to_dict()
    assert counts == {"ok": 100, "request_failed": 100, "not_sent": 50}
    assert len(df) == 250 and df["row_number"].is_unique  # nothing dropped
    assert df.loc[df["status"] == "not_sent", "error_message"].str.contains("Not sent").all()


def test_http_error_on_one_chunk_does_not_stop_the_rest():
    p = batch.parse_csv(many_rows(250))
    out = batch.run_batch(recording_client(status_on_call={2: 422}), p.table)
    counts = batch.assemble(p.table, out)["status"].value_counts().to_dict()
    assert counts == {"ok": 150, "request_failed": 100}


def test_per_item_backend_errors_are_kept():
    p = batch.parse_csv(many_rows(3))
    out = batch.run_batch(recording_client(errors={1: {1: ("input_too_long", "too long")}}), p.table)
    df = batch.assemble(p.table, out)
    assert df["status"].tolist() == ["ok", "error", "ok"]
    assert df.loc[1, "error_code"] == "input_too_long" and df.loc[1, "error_message"] == "too long"


def test_zero_valid_rows_sends_nothing():
    seen = []
    p = batch.parse_csv(b"original_text,simplified_text\n,B\n")
    out = batch.run_batch(recording_client(record=seen), p.table)
    assert seen == [] and out == {} and batch.assemble(p.table, out)["status"].tolist() == ["invalid_input"]


# ------------------------------------------------------------------ result content: no fake scores
def test_model_unavailable_gives_empty_score_columns_but_real_findings():
    p = batch.parse_csv(many_rows(3))
    df = batch.assemble(p.table, batch.run_batch(recording_client(available=False), p.table))
    assert df["predicted_lmp_score"].isna().all() and (df["preservation_category"] == "").all()
    assert (df["lmp_status"] == "unavailable").all() and (df["n_findings"] == 1).all()
    assert df.loc[0, "finding_types"] == "obligation_permission_change" and "[strong]" in df.loc[0, "findings"]
    s = batch.summarize(df)
    assert s["with_lmp_score"] == 0 and s["lmp_unavailable"] == 3 and s["ok"] == 3


def test_model_available_controlled_fixture_scores_are_passed_through():
    p = batch.parse_csv(many_rows(2))
    df = batch.assemble(p.table, batch.run_batch(recording_client(available=True), p.table))
    assert df["predicted_lmp_score"].tolist() == [8.0, 8.0] and (df["preservation_category"] == "Preserved").all()
    assert batch.summarize(df)["with_lmp_score"] == 2


def test_assemble_never_shows_a_score_for_an_unavailable_payload():
    p = batch.parse_csv(many_rows(1))
    bad = fb.predict_response(False)
    bad["predicted_lmp_score"] = 9.9  # a payload the client would normally reject; assemble must still not display it
    df = batch.assemble(p.table, {1: ("ok", bad)})
    assert pd.isna(df.loc[0, "predicted_lmp_score"])


def test_baseline_columns_only_when_available():
    p = batch.parse_csv(many_rows(1))
    payload = fb.predict_response(False, requested=True, baselines={
        "tfidf": {"status": "available", "score": 6.5, "raw_similarity": 0.8},
        "multilingual_embedding": {"status": "unavailable", "score": None, "reason": "not_implemented"}})
    df = batch.assemble(p.table, {1: ("ok", payload)})
    assert df.loc[0, "baseline_tfidf_score"] == 6.5 and pd.isna(df.loc[0, "baseline_multilingual_embedding_score"])


# ------------------------------------------------------------------ downloadable results
def test_export_is_a_faithful_csv_with_every_row_and_id():
    rows = ["id0,Good one.,Good two.,en,,\n", "id1,,Bad row.,en,,\n", "id2,Third.,Third b.,en,,\n"]
    p = batch.parse_csv(csv_bytes(rows))
    df = batch.assemble(p.table, batch.run_batch(recording_client(), p.table))
    raw = batch.to_csv_bytes(df)
    assert raw.startswith(b"\xef\xbb\xbf")  # BOM for Excel
    back = pd.read_csv(io.BytesIO(raw), dtype=str, keep_default_na=False)
    assert back["row_number"].tolist() == ["1", "2", "3"] and back["input_id"].tolist() == ["id0", "id1", "id2"]
    assert back["status"].tolist() == ["ok", "invalid_input", "ok"]
    assert {"predicted_lmp_score", "preservation_category", "lmp_status", "finding_types", "findings", "error_message"} <= set(back.columns)
    assert (back["predicted_lmp_score"] == "").all()  # model unavailable -> empty, never a placeholder number


def test_export_neutralises_spreadsheet_formulas():
    p = batch.parse_csv(b"original_text,simplified_text\n=1+1,@SUM(A1)\n-5 days,+cmd\n")
    df = batch.assemble(p.table, batch.run_batch(recording_client(), p.table))
    back = pd.read_csv(io.BytesIO(batch.to_csv_bytes(df)), dtype=str, keep_default_na=False)
    assert back["original_text"].tolist() == ["'=1+1", "'-5 days"] and back["simplified_text"].tolist() == ["'@SUM(A1)", "'+cmd"]
    assert df["original_text"].tolist()[0] == "=1+1"  # the in-app table keeps the original text


def test_example_csv_in_the_repo_is_valid_apart_from_its_two_deliberate_bad_rows():
    from pathlib import Path
    p = batch.parse_csv((Path(__file__).resolve().parents[1] / "data" / "sample" / "batch_example.csv").read_bytes())
    assert p.file_errors == [] and len(p.table) == 6 and sorted({e.row_number for e in p.row_errors}) == [5, 6]


def test_full_pipeline_against_the_real_backend_in_process(tmp_path):
    """Group: REAL backend code (in-process). No model installed, so scores must stay empty."""
    from app.config import Settings
    from app.main import create_app
    from fastapi.testclient import TestClient
    st = Settings(model_dir=tmp_path / "m", baseline_dir=tmp_path / "b", results_dir=tmp_path / "r")
    with TestClient(create_app(st)) as tc:
        def handler(request):
            r = tc.request(request.method, request.url.path, content=request.content, headers={"content-type": "application/json"})
            return httpx.Response(r.status_code, content=r.content, headers={"content-type": "application/json"})
        client = ApiClient("http://in-process.test", transport=httpx.MockTransport(handler))
        p = batch.parse_csv(many_rows(130))
        df = batch.assemble(p.table, batch.run_batch(client, p.table))
    assert len(df) == 130 and (df["status"] == "ok").all() and df["predicted_lmp_score"].isna().all()
    assert (df["finding_types"] == "obligation_permission_change").all()


@pytest.mark.live_server
def test_batch_over_real_http_to_a_live_server(live_url):
    """REAL backend, REAL HTTP (uvicorn subprocess): 250 rows -> 3 requests; no model installed so scores stay empty."""
    p = batch.parse_csv(many_rows(250), ["en"])
    df = batch.assemble(p.table, batch.run_batch(ApiClient(live_url), p.table))
    assert len(df) == 250 and (df["status"] == "ok").all() and df["predicted_lmp_score"].isna().all()
    assert (df["lmp_status"] == "unavailable").all() and (df["finding_types"] == "obligation_permission_change").all()
