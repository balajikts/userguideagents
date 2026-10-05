from scripts.ingest_manuals import build_chunks, chunk_text


def test_chunking_respects_size_and_overlap():
    text = "\n\n".join(f"Paragraph {i}. " + "word " * 40 for i in range(20))
    chunks = chunk_text(text, size=500, overlap=100)
    assert len(chunks) > 3
    assert all(len(c) <= 750 for c in chunks)
    assert chunks[1][:50] in chunks[0][-150:] or chunks[0][-100:] in chunks[1]


def test_giant_paragraph_is_split():
    chunks = chunk_text("x" * 5000, size=900, overlap=150)
    assert len(chunks) >= 5 and all(len(c) <= 1350 for c in chunks)


def test_build_chunks_pages_and_filters_tiny():
    pages = ["Reset: hold power and custom buttons for 7 seconds to initialize the headset.", "p2", ""]
    chunks = build_chunks(pages, brand="Sony", model="WH-1000XM5", device_type="headphones", url="u", title="t")
    assert len(chunks) == 1 and chunks[0].page == 1
