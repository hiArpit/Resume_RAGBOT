from typing import List
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# RecursiveCharacterTextSplitter is the main chunking tool from LangChain
# Split long text into piece("chunks")
# Make sure that the chunks are meaningful
# overlap being done between chunks to prevent lost context

from langchain_core.documents import Document
# langchain own way to store text and metadata together(text:- actual text and metadata:- data about text such as page no. and source of the tex)

def make_chunks(
    pages_text: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> List[Document]: 
    # chunk_size:- maximum characters in each chunk
    # chunk_overlap:- how many characters new chunk share with the previous one
    # The function will return a list of Document objects, each object will represent chunks

    chunks: List[Document] = []
    # Empty list of Documents

    for i, page_text in enumerate(pages_text):
        # Loop through each page
        if not page_text:
            continue
            # If the page is empty -> skip

        # Clean up page text a bit
        text = page_text.strip()

        start = 0
        text_length = len(text)

        # Sliding window over the text
        while(start < text_length):
            end = start + chunk_size
            chunk_text = text[start:end]

            # Create a document for this chunk
            doc = Document(
                    page_content=chunk_text,
                    metadata={"page": i+1, "source": "Arpit_Negi_Resume.pdf"},
            )
            # Creating a document that contains chunk content
            # Metadata contains page number and source
            chunks.append(doc)

            start += chunk_size - chunk_overlap
    
    return chunks

    #     # Creating the chunk splitter
    #     splitter = RecursiveCharacterTextSplitter(
    #         chunk_size=chunk_size,
    #         chunk_overlap=chunk_overlap,
    #     )
    #     # RecursiveCharacterTextSplitter preserves meaning while splitting
    #     # avoids breaking in weird places
    #     # ensures chunks aren't too long
    #     # ensures small overlap between chunks so context will not be lost

    # chunked_docs = splitter.split_documents(docs)
    # # splitter takes your docs(list of documents)
    # # Split their page_content, but presevers their metadata
    # # returns many smaller documents

    # return chunked_docs
    # # This output will go for embeddings, Vector DB(Chroma) and be searched during Retrieval


if __name__ == "__main__":
    # When this file be run directly, this test block will run 
    import os
    from pdf_loader import load_pdf_pages

    pdf_path = os.path.join("data", "Arpit_Negi_Resume.pdf")
    # Path to pdf 
    pages = load_pdf_pages(pdf_path)
    # returns the list of page texts
    chunks = make_chunks(pages)
    # returns the splitted chunks

    print(f"Total pages: {len(pages)}")
    print(f"Total chunks: {len(chunks)}")
    # How many pages and chunks are produced
    print("\nExample chunk:")
    if chunks:
        print("Metadata:", chunks[0].metadata)
        # Metadata of first chunk
        print("Content:\n", chunks[0].page_content[:500])
        # first 500 characters of first chunk