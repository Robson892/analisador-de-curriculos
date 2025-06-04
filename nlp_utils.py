import re
import spacy
from spacy.pipeline import EntityRuler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from wordcloud import WordCloud
import matplotlib.pyplot as plt


# ------------- REGEX BÁSICOS ------------------------------------------------ #
EMAIL_RE   = re.compile(r"[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+")
PHONE_RE   = re.compile(r"(\(?\d{2}\)?[\s-]?)?9?\d{4}[-\s]?\d{4}")


# ------------- FUNÇÕES DE EXTRAÇÃO BÁSICA ----------------------------------- #
def extrair_email(texto: str):
    """Retorna o primeiro e-mail encontrado ou None."""
    emails = EMAIL_RE.findall(texto)
    return emails[0] if emails else None


def extrair_telefone(texto: str):
    """Retorna o primeiro telefone encontrado ou None."""
    tels = PHONE_RE.findall(texto)
    return tels[0] if tels else None


def extrair_nome(doc):
    """
    Tenta retornar a primeira entidade Pessoa reconhecida;
    se falhar, devolve a primeira linha do texto como fallback.
    """
    for ent in doc.ents:
        if ent.label_ in ("PER", "PERSON"):
            return ent.text
    return doc.text.strip().split("\n")[0].strip()


def extrair_skills(doc):
    """Coleta todas as entidades rotuladas como SKILL."""
    return list({ent.text for ent in doc.ents if ent.label_ == "SKILL"})


# ------------- EXTRAÇÃO DE SKILLS VIA SEÇÃO DE TEXTO ------------------------- #
def extrair_skills_secao(texto):
    """
    Busca seção no texto que contenha título 'habilidades', 'competências',
    'skills' ou 'conhecimentos' e extrai os termos listados logo após.
    Retorna lista de skills encontradas na seção (pode ser vazia).
    """
    padrao_secao = re.compile(r'(habilidades|competências|skills|conhecimentos)\s*[:\-]?', re.IGNORECASE)
    match = padrao_secao.search(texto)
    if not match:
        return []

    inicio = match.end()
    # Pega um trecho curto após o título (ajuste tamanho se quiser)
    trecho = texto[inicio:inicio + 300]  

    # Quebra o trecho por vírgula, ponto-e-vírgula ou nova linha
    candidatos = re.split(r'[,\n;]', trecho)

    # Limpa espaços e filtra itens vazios
    skills = [item.strip() for item in candidatos if item.strip()]

    return skills


# ------------- PIPELINE DINÂMICO COM SKILLS ---------------------------------- #
def criar_pipeline_com_skills(skills_list):
    """
    Cria e devolve um objeto `nlp` contendo um EntityRuler com as
    skills passadas pelo usuário.
    """
    nlp = spacy.load("pt_core_news_sm")

    # Adiciona EntityRuler apenas se ainda não existir
    if "entity_ruler" not in nlp.pipe_names:
        nlp.add_pipe("entity_ruler", before="ner")

    ruler: EntityRuler = nlp.get_pipe("entity_ruler")

    # Limpa padrões antigos do ruler para evitar sobreposição
    ruler.clear()
    patterns = [{"label": "SKILL", "pattern": skill.strip()} for skill in skills_list if skill.strip()]
    ruler.add_patterns(patterns)
    return nlp


# ------------- PROCESSAMENTO COMPLETO DE TEXTO ------------------------------ #
def processar_texto_com_pipeline(texto: str, nlp):
    """
    Executa o pipeline spaCy e retorna um dicionário com:
    nome, telefone, e-mail e lista de skills identificadas,
    incluindo skills extraídas pela seção explícita no texto.
    """
    doc = nlp(texto)
    skills_ner = set(extrair_skills(doc))
    skills_secao = set(extrair_skills_secao(texto))

    # Une e ordena alfabeticamente para consistência
    skills_unificadas = sorted(skills_ner.union(skills_secao))

    return {
        "nome": extrair_nome(doc),
        "telefone": extrair_telefone(texto),
        "email": extrair_email(texto),
        "skills": skills_unificadas,
    }


# ------------- SIMILARIDADE VAGA × CURRÍCULO -------------------------------- #
def calcular_similaridade(texto_vaga: str, texto_cv: str) -> float:
    """
    Calcula similaridade TF-IDF + cosseno (0-1) entre a descrição
    da vaga e o currículo.
    """
    vetorizador = TfidfVectorizer().fit_transform([texto_vaga, texto_cv])
    sim = cosine_similarity(vetorizador[0:1], vetorizador[1:2])
    return float(sim[0][0])


# ------------- AGRUPAMENTO (ALTA / MÉDIA / BAIXA) --------------------------- #
def agrupar_aderencia(porcentagem):
    if porcentagem >= 80:
        return "Alta"
    elif porcentagem >= 50:
        return "Média"
    else:
        return "Baixa"


# ------------- NUVEM DE PALAVRAS ------------------------------------------- #
def gerar_nuvem_palavras(textos, max_palavras: int = 100):
    """
    Recebe lista de strings, gera e retorna um objeto WordCloud.
    """
    corpus = " ".join(textos)
    return WordCloud(
        width=800,
        height=400,
        background_color="white",
        max_words=max_palavras,
        colormap="viridis"
    ).generate(corpus)


def plotar_nuvem(wc: WordCloud):
    """
    Recebe um WordCloud e devolve uma figura matplotlib pronta
    para ser exibida no Streamlit.
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wc, interpolation="bilinear")
    ax.axis("off")
    plt.tight_layout()
    return fig
