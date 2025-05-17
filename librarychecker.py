from langchain_unstructured import UnstructuredLoader

loader = UnstructuredLoader(r"Design through Verilog HDL  by Padmanabhan.pdf")
try:
    documents = loader.load()
    print("Document content:", documents)
except Exception as e:
    print("Error:", e)
