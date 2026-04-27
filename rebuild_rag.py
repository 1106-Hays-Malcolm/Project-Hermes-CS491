from rag.rag_api import RAGAPI

rag = RAGAPI()

# Clear manifest so files are allowed to ingest again
rag.clear_manifest()

# Re-ingest cleaned JSON files
rag.force_reingest("./data/json", "quests")

# Test query
results = rag.query("test", n_per_collection=1)

for r in results:
    print(r["metadata"])
    