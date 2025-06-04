import fitz  # PyMuPDF
import re
import spacy

nlp = spacy.load("pt_core_news_sm")

def extrair_texto(pdf_file):
    with fitz.open(stream=pdf_file.read(), filetype="pdf") as doc:
        texto = ""
        for page in doc:
            texto += page.get_text()
    return texto

def extrair_nome_spacy(texto):
    doc = nlp(texto)
    nomes = [ent.text for ent in doc.ents if ent.label_ == "PER"]
    if nomes:
        return nomes[0].strip()
    else:
        linhas = texto.split('\n')
        for linha in linhas:
            if len(linha.strip().split()) >= 2 and not re.search(r'\d', linha):
                return linha.strip()
    return "Nome não identificado"

def extrair_telefone(texto):
    padrao_telefone = re.compile(
        r'(\+?55\s?)?(\(?\d{2}\)?[\s-]?)?(\d{4,5}[\s-]?\d{4})'
    )
    telefones = padrao_telefone.findall(texto)
    if telefones:
        for tel_parts in telefones:
            telefone = ''.join(tel_parts).strip()
            telefone = re.sub(r'\s+', '', telefone)
            telefone = telefone.replace('-', '')
            if len(telefone) >= 8:
                return telefone
    return "Telefone não identificado"

def extrair_info_contato(texto):
    email = re.search(r'[\w\.-]+@[\w\.-]+', texto)
    telefone = extrair_telefone(texto)
    return {
        "E-mail": email.group() if email else "",
        "Telefone": telefone
    }
