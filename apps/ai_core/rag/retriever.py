from .indexer import RAGIndex
_index = None

def get_index():
    global _index
    if _index is None:
        _index = RAGIndex()
    return _index

def ingest_example_docs():
    """Call once to seed with a few pages (replace with your own)."""
    idx = get_index()
    docs = [
        {"title":"AWS S3 Overview","text":"Amazon S3 is object storage with scalability, data availability, security, and performance...", "source":"aws-docs:s3"},
        {"title":"Amazon EBS vs EFS","text":"EBS is block storage for EC2. EFS is elastic file storage for Linux EC2. S3 is object...", "source":"aws-docs:storage"},
        {"title":".NET Delegates","text":"A delegate is a type that represents references to methods with a particular parameter list and return type...", "source":"ms-docs:delegates"},
        {"title":"SQL GROUP BY","text":"GROUP BY groups rows sharing a property so aggregate functions can be applied to each group...", "source":"sql:groupby"},
    ]
    idx.add_documents(docs)
