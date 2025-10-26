import os
from googleapiclient.discovery import build
import pandas as pd
from transformers import pipeline
from groq import Groq
import json
import time
from dotenv import load_dotenv

load_dotenv()

YOUTUBE_API_KEY = os.getenv('YOUTUBE_API_KEY')

GROQ_API_KEY = os.getenv('GROQ_API_KEY')

groq_client = Groq(api_key=GROQ_API_KEY)

#tentando rodar o sentiment model
sentiment_model = pipeline(
"sentiment-analysis",
    model="cardiffnlp/twitter-roberta-base-sentiment-latest"
)
print('modelo sentimento carregado')

#coleta dos dados no yt

def buscar_comentarios(jogador, max_videos=10, max_comentarios=100):

    youtube = build('youtube', 'v3', developerKey=YOUTUBE_API_KEY)

    videos_processados = 0

    queries = [
        f"{jogador}",
        f"{jogador} nfl",
        f"{jogador} highlights",
        f"{jogador} 2024"
    ]

    for query in queries:
        if videos_processados >= max_videos:
            break

        videos = youtube.search().list(
            q=query,
            type='video',
            part='id',
            maxResults=max_videos * 2,
            order='relevance'
        ).execute()

    comentarios = []

    for video in videos['items']:
        video_id = video['id'] ['videoId']

        try:
            video_info = youtube.videos().list(
                id=video_id,
                part='snippet'
            ).execute()

            video_titulo = video_info['items'][0]['snippet']['title']
            video_data = video_info['items'][0]['snippet']['publishedAt']

            resultado = youtube.commentThreads().list(
                videoId=video_id,
                part='snippet',
                maxResults=max_comentarios,
                order='relevance'
            ).execute()

            for item in resultado ['items']:
                c = item['snippet']['topLevelComment']['snippet']

                comentarios.append({
                    'texto': c['textDisplay'],
                    'likes': c['likeCount'],
                    'video_id': video_id,
                    'video_titulo': video_titulo,
                    'video_publicado_em': video_data,
                    'comentario_publicado_em': c['publishedAt']
                })

        except Exception as e:
            print(f"erro {e}")
            continue

    return comentarios


def limpeza_groq(comentarios, jogador):
    textos = "\n".join([
        f"{i+1}. {c['texto'][:150]}"
        for i, c in enumerate(comentarios[:100])
    ])

    prompt = f""""Você é um analista de popularidade da NFL. Analise comentários sobre {jogador}:

COMENTÁRIOS:
{textos}
    
Para cada comentário válido (ignore irrelevantes/spam), retorne JSON:
{{"comentarios": [
    {{"num": 1, "texto": "texto limpo", "categoria": "desempenho"}},
    {{"num": 5, "texto": "texto limpo", "categoria": "habilidade"}}
]}}

Categorias: desempenho, habilidade, lesao, comparacao, outro
    
retorne apenas o JSON, nada mais

REGRAS CRÍTICAS:
1. NO CAMPO TEXTO: Use apenas aspas simples (') ou remova aspas completamente
2. Nunca use aspas duplas (") dentro do texto
3. Remova emojis e caracteres especiais"""

    try:
        resposta = groq_client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.3,
            max_tokens=2000
        )

        conteudo = resposta.choices[0].message.content

        inicio = conteudo.find('{')
        fim = conteudo.rfind('}') + 1
        json_str = conteudo[inicio:fim]

        try:
            print(f" resultado da limpeza do agente: {json_str[:300]}")
            dados = json.loads(json_str)
        except json.JSONDecodeError:
            print(f" JSON com erro, tentando limpar...")
            json_limpo = json_str.replace('\n', ' ').replace('\r', '')
            try:
                dados = json.loads(json_limpo)
            except:
                raise

        limpos = []
        for item in dados['comentarios']:
            try:
                idx = item['num'] - 1
                if idx < len(comentarios):
                    limpos.append({
                        'texto_limpo': item['texto'],
                        'categoria': item['categoria'],
                        'likes': comentarios[idx]['likes'],
                        'video_id': comentarios[idx]['video_id'],
                        'video_titulo': comentarios[idx]['video_titulo'],
                        'video_publicado_em': comentarios[idx]['video_publicado_em'],
                        'comentario_publicado_em': comentarios[idx]['comentario_publicado_em']
                    })
            except:
                continue
        if limpos:
            print(f"\n [{jogador}] Groq processou {len(limpos)} comentários")
            print(f"Exemplo: '{limpos[0]['texto_limpo'][:200]}...' → {limpos[0]['categoria']}")
            # =========================

        return limpos

    except Exception as e:
        print(f"erro no groq {e}")

        return [{
            'texto_limpo': c['texto'][:200],
            'categoria': 'outro',
            'likes': c['likes'],
            'video_id': c['video_id'],
            'video_titulo': c['video_titulo'],
            'video_publicado_em': c['video_publicado_em'],
            'comentario_publicado_em': c['comentario_publicado_em']
        } for c in comentarios [:200]]

#hugging face
def analisar_sentimento(comentarios):

    resultados = []
    total = len(comentarios)

    for i, c in enumerate(comentarios):
        try:
            texto = c['texto_limpo'][:512]
            resultado = sentiment_model(texto)[0]

            label = resultado['label'].lower()

            if 'negative' in label:
                sent = 'Negativo'
                valor = 1
            elif 'positive' in label:
                sent = 'Positivo'
                valor = 5
            else:
                sent = 'Neutro'
                valor = 3

            resultados.append({
                'texto': c['texto_limpo'],
                'categoria': c['categoria'],
                'sentimento': sent,
                'score': valor,
                'confianca': round(resultado['score'], 2),
                'likes': c['likes'],
                'video_id': c['video_id'],
                'video_titulo': c['video_titulo'],
                'video_publicado_em': c['video_publicado_em'],
                'comentario_publicado_em': c['comentario_publicado_em']
            })

        except Exception as e:
            continue

    return resultados

#data frame no pandas
def criar_dataframe(resultados, jogador):

    df = pd.DataFrame(resultados)
    df['jogador'] = jogador
    df['engajamento'] = df['likes'] * df['score']

    colunas = [
        'jogador', 'sentimento', 'score', 'confianca', 'categoria',
        'texto', 'likes', 'engajamento', 'video_id', 'video_titulo',
        'video_publicado_em', 'comentario_publicado_em'
    ]

    return df[colunas].sort_values('engajamento', ascending=False)

#mini estruturacao
def estruturar_para_powerbi(lista_dataframes, arquivo="Qb_vs_RB.csv"):

    df = pd.concat(lista_dataframes, ignore_index=True)

    df['video_publicado_em'] = pd.to_datetime(df['video_publicado_em']).dt.tz_localize(None)
    df['comentario_publicado_em'] = pd.to_datetime(df['comentario_publicado_em'])

    df['video_ano'] = df['video_publicado_em'].dt.year
    df['video_mes'] = df['video_publicado_em'].dt.month
    df['video_mes_nome'] = df['video_publicado_em'].dt.month_name()
    df['dias_desde_publicacao'] = (pd.Timestamp.now() - df['video_publicado_em']).dt.days

    df['nivel_engajamento'] = pd.cut(
        df['engajamento'],
        bins=[0, 10, 50, float('inf')],
        labels=['Baixo', 'Médio', 'Alto']
    )

    df['idade_video'] = pd.cut(
        df['dias_desde_publicacao'],
        bins=[0, 7, 30, 90, float('inf')],
        labels=['Última Semana', 'Último Mês', 'Últimos 3 Meses', 'Mais Antigo']
    )

    df.to_csv(arquivo, index=False, encoding='utf-8-sig')

    print("\n SENTIMENTO MÉDIO POR JOGADOR:")
    print(df.groupby('jogador')['score'].mean().round(2))

    print("\n DISTRIBUIÇÃO GERAL:")
    print(df['sentimento'].value_counts())

    return df

#rodar tudo
def analisar(jogador, max_videos=10, max_comentarios=100):

    comentarios = buscar_comentarios(jogador, max_videos=max_videos, max_comentarios=max_comentarios)

    if not comentarios:
        print("nenhum comentario")
        return None

    limpos = limpeza_groq(comentarios, jogador)

    resultados = analisar_sentimento(limpos)

    if not resultados:
        print("nenhum resultado dps da analise")
        return None

    df = criar_dataframe(resultados, jogador)


    return df

#exe

if __name__ == "__main__":

    jogadores = [
        "Runningback",
        "Quarterback"
    ]

    todos_resultados = []

    for jogador in jogadores:
        df = analisar(jogador, max_videos=10, max_comentarios=100)
        if df is not None and len(df) > 0:
            todos_resultados.append(df)
        time.sleep(2)

    if todos_resultados:
        df_final = estruturar_para_powerbi(todos_resultados)
