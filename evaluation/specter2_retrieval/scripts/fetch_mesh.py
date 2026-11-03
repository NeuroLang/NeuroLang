import json, time, urllib.request, urllib.parse, xml.etree.ElementTree as ET
ids = json.load(open("/tmp/stage3/ids.json"))
URL = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi"
mesh = {}
t0 = time.time()
for n in range(0, len(ids), 200):
    chunk = ids[n:n + 200]
    data = urllib.parse.urlencode({"db": "pubmed", "id": ",".join(chunk), "retmode": "xml", "tool": "neurolang-eval"}).encode()
    for attempt in range(4):
        try:
            with urllib.request.urlopen(urllib.request.Request(URL, data=data), timeout=120) as r:
                root = ET.fromstring(r.read()); break
        except Exception as e:
            print("retry", n, e, flush=True); time.sleep(2 ** attempt)
    for art in root.iter("PubmedArticle"):
        pmid = art.findtext("MedlineCitation/PMID")
        mesh[pmid] = [(d.text, d.get("UI"), d.get("MajorTopicYN")) for d in art.iterfind("MedlineCitation/MeshHeadingList/MeshHeading/DescriptorName")]
    if (n // 200) % 10 == 0: print(n, len(mesh), round(time.time() - t0), "s", flush=True)
    time.sleep(0.4)
json.dump(mesh, open("/tmp/stage3/mesh.json", "w"))
print("done", len(mesh), "with mesh:", sum(1 for v in mesh.values() if v), flush=True)
