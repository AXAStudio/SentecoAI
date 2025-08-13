import io, zipfile, unicodedata

def normalize_text(s):
    repl = {'\u2019':"'", '\u2018':"'", '\u201c':'"', '\u201d':'"', '\u2013':'-', '\u2014':'-', '\u00a0':' '}
    return unicodedata.normalize("NFKC", s.translate(str.maketrans(repl)))


from app.config import MODEL_VARIANTS as srcs

src_format = "app/models/model_{}.keras"
dst = "app/models/model_{}_utf8.keras"

for src in srcs:
    with zipfile.ZipFile(
        src_format.format(src), "r"
    ) as zin, zipfile.ZipFile(
        dst.format(src), "w", zipfile.ZIP_DEFLATED
    ) as zout:
        for info in zin.infolist():
            data = zin.read(info.filename)
            if info.filename.startswith("assets/") and info.filename.endswith(".txt"):
                try:
                    text = data.decode("utf-8")
                except UnicodeDecodeError:
                    text = data.decode("cp1252")
                text = "\n".join(normalize_text(line) for line in text.splitlines())
                data = text.encode("utf-8")
            zout.writestr(info, data)
