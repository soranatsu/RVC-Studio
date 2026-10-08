def read_text(path, errors="strict", newline=None):
    last_error = None
    for encoding in ("utf-8-sig", None, "gbk"):
        try:
            kwargs = {"errors": "strict", "newline": newline}
            if encoding is not None:
                kwargs["encoding"] = encoding
            with open(path, "r", **kwargs) as file:
                return file.read()
        except UnicodeDecodeError as error:
            last_error = error
    if errors != "strict":
        with open(path, "r", encoding="gbk", errors=errors, newline=newline) as file:
            return file.read()
    raise last_error


def read_faiss_index(path):
    import os
    import faiss

    path = os.fsdecode(path)
    if path.isascii():
        return faiss.read_index(path)
    with open(path, "rb") as stream:
        return faiss.read_index(faiss.PyCallbackIOReader(stream.read))
