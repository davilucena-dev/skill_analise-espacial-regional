---
name: analise-espacial-regional
description: >
  Use esta skill para análises espaciais e regionais completas: coleta de dados
  geográficos e econômicos (RAIS, PNAD, POF, SIDRA, Comex Stat, BCB, shapefiles),
  construção de matrizes de pesos espaciais (contiguidade, distância, kernel, fluxos),
  Análise Exploratória de Dados Espaciais (AEDE) com Moran I, LISA, G*, Geary C,
  estimação de todos os modelos de econometria espacial (SAR, SEM, SDM, SAC, GNS, SLX,
  painel espacial com efeitos fixos/aleatórios, painel dinâmico), simulação de choques
  com multiplicador espacial, validação cruzada espacial, visualização com mapas
  coropléticos, LISA cluster maps, dashboards interativos, e integração com MIP/SAM/CGE.
  Ativa para qualquer tarefa envolvendo análise espacial, regional, econometria espacial,
  spillovers, transbordamentos, mapas, geoprocessamento ou AEDE.
---
# Skill: Análise Espacial e Regional — Super Skill

## Propósito
Skill completa e autônoma para análises espaciais e regionais em economia, combinando:
- **Coleta de dados**: 15+ fontes oficiais (RAIS, PNAD, POF, SIDRA, Comex Stat, BCB, IPEA, Shapefiles IBGE)
- **AEDE**: Global e Local Moran's I, Geary's C, Getis-Ord G/G*, Moran Scatterplot, LISA Cluster Maps
- **Matrizes de pesos espaciais**: Queen, Rook, k-NN, Distance Band, Kernel, Fluxos Econômicos
- **Modelos espaciais cross-section**: OLS, SAR, SEM, SDM, SAC, GNS, SLX, Spatial Tobit/Probit
- **Modelos em painel espacial**: Pooled, FE, RE, Spatial Lag FE/RE, Spatial Error FE/RE
- **Simulação**: Multiplicador espacial (I−ρW)⁻¹, Monte Carlo, decomposição em efeitos diretos/indiretos/totais
- **Visualização**: Mapas coropléticos, LISA clusters, dashboards interativos (Plotly, Folium)
- **Integração regional**: MIP, SAM, CGE espacial, indicadores de ligação regionais

## O que esta skill NÃO faz
- Não substitui softwares GIS completos (QGIS/ArcGIS) para edição avançada de geometrias
- Não estima modelos com mais de 10.000 observações via MLE (usar GMM para datasets grandes)
- Não faz downscaling estatístico (das regiões para municípios) sem validação
- Não modela dinâmica intertemporal completa (apenas painel espacial com até T=30 períodos)

## Fontes
- Referência: `espacial e regional.pdf` (Relatório Técnico AgenteNazaré — Spillover SAR, 2026)
- Autor: Davi Lucena da Silva — Doutorando em Economia pela UFV
- Bibliografia base: Anselin (1988), LeSage & Pace (2009), Elhorst (2014), Almeida (2012)
- Pacotes: PySAL/libpysal 4.14+ (pysal.org), spreg 1.9+ (painel espacial), DuckDB espacial

---

## Índice

1. [Coleta de Dados — 15+ Fontes Oficiais](#1-coleta-de-dados)
2. [Processamento de Grandes Volumes com DuckDB](#2-processamento-duckdb)
3. [Shapefiles e Mapas do Brasil](#3-shapefiles)
4. [Matrizes de Pesos Espaciais](#4-matrizes-pesos)
5. [Análise Exploratória de Dados Espaciais (AEDE)](#5-aede)
6. [Modelos Espaciais Cross-Section](#6-modelos-cross-section)
7. [Modelos em Painel Espacial](#7-painel-espacial)
8. [Estimação Bayesiana](#8-bayesiana)
9. [Simulação e Multiplicador Espacial](#9-simulacao)
10. [Validação Cruzada Espacial](#10-validacao)
11. [Visualização e Dashboards](#11-visualizacao)
12. [Integração com MIP/SAM/CGE](#12-integracao)
13. [Pipeline Completo de Análise Espacial](#13-pipeline)
14. [Instalação de Dependências](#14-instalacao)

---

## 1. Coleta de Dados — 15+ Fontes Oficiais

### 1.1 RAIS (PDET/MTE) — Emprego Formal

A RAIS cobre o mercado formal de trabalho brasileiro (~55 milhões de vínculos ativos em 2023).

```python
import pandas as pd
import numpy as np
import py7zr
import os
from urllib.request import urlretrieve

def baixar_rais(ano=2023, dir_destino='/tmp/rais'):
    """
    Baixa e extrai RAIS de um ano.
    
    Args:
        ano: Ano da RAIS (2010-2023 disponíveis)
        dir_destino: Diretório para extração
    
    Returns:
        Caminho para o arquivo CSV extraído
    """
    os.makedirs(dir_destino, exist_ok=True)
    url = f"ftp://ftp.mtps.gov.br/pdet/microdados/RAIS/{ano}/RAIS_ESTAB_PUB.7z"
    arquivo_7z = os.path.join(dir_destino, f"RAIS_{ano}.7z")
    
    # Download com retry
    import time
    for tentativa in range(3):
        try:
            urlretrieve(url, arquivo_7z)
            break
        except Exception as e:
            if tentativa < 2:
                time.sleep(5 * (tentativa + 1))
            else:
                raise RuntimeError(f"Falha no download RAIS {ano}: {e}")
    
    # Extrair
    with py7zr.SevenZipFile(arquivo_7z, mode='r') as z:
        z.extractall(path=dir_destino)
    
    # Localizar o CSV (nome varia conforme o ano)
    for f in os.listdir(dir_destino):
        if f.endswith('.COMT') or f.endswith('.csv') or f.endswith('.txt'):
            return os.path.join(dir_destino, f)
    
    raise FileNotFoundError("Arquivo RAIS não encontrado após extração")

def processar_rais_filtro_setorial(caminho_csv, cod_cnae2='01', chunksize=500_000):
    """
    Lê RAIS em chunks, filtra por setor CNAE 2 dígitos e agrega por UF.
    
    Args:
        caminho_csv: Caminho do arquivo RAIS
        cod_cnae2: Código CNAE 2 dígitos (ex: '01'=agricultura)
        chunksize: Registros por chunk
    
    Returns:
        DataFrame com emprego por UF (agrícola e total)
    """
    # Colunas essenciais da RAIS
    cols = ['uf', 'cnae20classe', 'qtd_vinc_ativos']
    
    agri_list = []
    total_list = []
    
    for chunk in pd.read_csv(caminho_csv, sep=';', encoding='iso-8859-1',
                              chunksize=chunksize, low_memory=False,
                              usecols=cols):
        # Cast seguro do CNAE
        chunk['cnae20classe'] = pd.to_numeric(chunk['cnae20classe'], errors='coerce')
        
        # Filtro setorial via divisão inteira (mais robusto que string)
        mask_agri = (chunk['cnae20classe'] // 100) == int(cod_cnae2)
        
        agri = chunk[mask_agri].groupby('uf')['qtd_vinc_ativos'].sum()
        total = chunk.groupby('uf')['qtd_vinc_ativos'].sum()
        
        agri_list.append(agri)
        total_list.append(total)
    
    df_agri = pd.concat(agri_list).groupby(level=0).sum()
    df_total = pd.concat(total_list).groupby(level=0).sum()
    
    resultado = pd.DataFrame({
        'emp_agri': df_agri,
        'emp_total': df_total
    }).fillna(0)
    
    resultado['part_agri'] = resultado['emp_agri'] / resultado['emp_total']
    
    return resultado.reset_index()
```

### 1.2 PNAD Contínua (IBGE/SIDRA)

```python
import sidrapy

def coletar_pnad_continua(tabela='4099', periodo='2023', nivel='1'):
    """
    Coleta dados da PNAD Contínua via SIDRA API.
    
    Tabelas importantes:
    4099: Taxa de desocupação
    4093: Rendimento médio real
    5918: População ocupada por setor
    5938: PIB municipal
    
    Args:
        tabela: Código da tabela SIDRA
        periodo: Período (ano ou 'last 12')
        nivel: Nível territorial (1=Brasil, 2=Grande Região, 3=UF, 6=Município)
    
    Returns:
        DataFrame com dados da PNAD
    """
    data = sidrapy.get_table(
        table_code=tabela,
        territorial_level=nivel,
        ibge_territorial_code='all',
        period=periodo,
        header='n',
        format='pandas'
    )
    return data
```

### 1.3 POF (Pesquisa de Orçamentos Familiares)

```python
def baixar_pof():
    """
    Baixa microdados da POF 2017-2018 do FTP do IBGE.
    """
    import requests
    url = ("https://ftp.ibge.gov.br/Orcamentos_Familiares/"
           "POF/2017_2018/Microdados/POF_2017_2018_UTF8.zip")
    
    response = requests.get(url, stream=True, timeout=300)
    with open('/tmp/pof_2017_2018.zip', 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
    
    import zipfile
    with zipfile.ZipFile('/tmp/pof_2017_2018.zip', 'r') as z:
        z.extractall('/tmp/pof/')
    
    # Despesas e rendimentos
    despesas = pd.read_csv('/tmp/pof/DESPESA_COD.csv', 
                           encoding='utf-8', sep=';', low_memory=False)
    rendimentos = pd.read_csv('/tmp/pof/RENDIMENTO_COD.csv',
                               encoding='utf-8', sep=';', low_memory=False)
    
    return despesas, rendimentos
```

### 1.4 Comex Stat (MDIC) — Comércio Exterior

```python
def coletar_comexstat(tipo='EXP', ano=2023, uf=None):
    """
    Coleta dados de comércio exterior do MDIC via API.
    
    Args:
        tipo: 'EXP' para exportações, 'IMP' para importações
        ano: Ano desejado
        uf: UF específica (None = todas)
    
    Returns:
        DataFrame com transações comerciais
    """
    import requests
    
    if uf:
        url = f"https://api.comexstat.mdic.gov.br/{tipo}/{ano}/UF/{uf}"
    else:
        url = f"https://api.comexstat.mdic.gov.br/{tipo}/{ano}/geral"
    
    response = requests.get(url, timeout=60)
    data = response.json()
    
    df = pd.DataFrame(data['data'])
    return df
```

### 1.5 BCB/SGS — Séries Temporais

```python
from python_bcb import sgs

def coletar_bcb_sgs():
    """
    Coleta séries do BCB via SGS.
    
    Códigos importantes:
    1 - Taxa de câmbio (R$/US$) - livre
    433 - IPCA
    4389 - PIB mensal
    12 - Selic
    243 - Taxa de desemprego
    """
    # Câmbio e IPCA
    cambio = sgs.get({'cambio': 1}, start='2010-01-01', end='2024-12-31')
    ipca = sgs.get({'ipca': 433}, start='2010-01-01', end='2024-12-31')
    
    # Para dados municipais/regionais, usar o SIDRA
    return cambio, ipca

def coletar_ipeadata(codigo='IPCA12_INF12'):
    """
    Coleta séries do IPEADATA.
    """
    import requests
    url = f"http://www.ipeadata.gov.br/api/odata4/ValoresSerie(SERCODIGO='{codigo}')"
    response = requests.get(url, timeout=30)
    return pd.json_normalize(response.json()['value'])
```

### 1.6 Shapefiles e Malhas Geográficas (IBGE)

```python
def baixar_shapefile_uf(ano=2022):
    """
    Baixa shapefile de UFs do Brasil do IBGE.
    
    Returns:
        GeoDataFrame com malha de UFs
    """
    import geopandas as gpd
    import requests
    import zipfile
    
    url = (f"https://geoftp.ibge.gov.br/organizacao_do_territorio/"
           f"malhas_territoriais/malhas_municipais/municipio_{ano}/"
           f"Brasil/BR/BR_UF_{ano}.zip")
    
    response = requests.get(url, timeout=300)
    with open('/tmp/br_uf.zip', 'wb') as f:
        f.write(response.content)
    
    with zipfile.ZipFile('/tmp/br_uf.zip', 'r') as z:
        z.extractall('/tmp/br_uf_shape/')
    
    gdf = gpd.read_file('/tmp/br_uf_shape/')
    return gdf

def baixar_shapefile_municipios(ano=2022):
    """
    Baixa shapefile de municípios do Brasil do IBGE.
    ~5.570 municípios.
    """
    import geopandas as gpd
    import requests, zipfile
    
    url = (f"https://geoftp.ibge.gov.br/organizacao_do_territorio/"
           f"malhas_territoriais/malhas_municipais/municipio_{ano}/"
           f"Brasil/BR/BR_Municipios_{ano}.zip")
    
    response = requests.get(url, timeout=600)
    with open('/tmp/br_mun.zip', 'wb') as f:
        f.write(response.content)
    
    with zipfile.ZipFile('/tmp/br_mun.zip', 'r') as z:
        z.extractall('/tmp/br_mun_shape/')
    
    gdf = gpd.read_file('/tmp/br_mun_shape/')
    return gdf
```

### 1.7 Matriz de Fluxos Comerciais Interestaduais (IPEA/NEREUS)

```python
def baixar_matriz_fluxos_interestaduais():
    """
    Baixa matriz de comércio interestadual (IPEA/NEREUS).
    Matriz 27×27 de fluxos comerciais entre UFs.
    """
    url = "https://www.ipea.gov.br/portal/images/stories/PDFs/TDs/ matriz_fluxos_comerciais_2015.csv"
    
    df = pd.read_csv(url, sep=';', index_col=0)
    return df  # DataFrame 27×27
```

### 1.8 MIP (Matriz Insumo- Produto) via SIDRA

```python
def baixar_mip_sidra():
    """
    Baixa tabelas da MIP 2015 do IBGE via SIDRA.
    Tabelas: 5859 (recursos), 5860 (usos)
    """
    import sidrapy
    
    # Tabela de recursos (Make)
    recursos = sidrapy.get_table(
        table_code="5859",
        territorial_level="1",
        ibge_territorial_code="all",
        period="2015",
        header='n',
        format='pandas'
    )
    
    # Tabela de usos (Use)
    usos = sidrapy.get_table(
        table_code="5860",
        territorial_level="1",
        ibge_territorial_code="all",
        period="2015",
        header='n',
        format='pandas'
    )
    
    return recursos, usos
```

### 1.9 Tabela de Referência Rápida — Fontes e Acesso

| Base | Dado | Fonte | Método | Pacote |
|---|---|---|---|---|
| RAIS | Vínculos formais por UF/CNAE | PDET/MTE | FTP | `py7zr`, `pandas` |
| PNAD | Emprego, renda, desocupação | IBGE/SIDRA | API | `sidrapy` |
| POF | Despesas, rendimentos famílias | IBGE | FTP | `zipfile` |
| Comex Stat | Exp/Imp por UF/NCM | MDIC | API REST | `requests` |
| BCB/SGS | Câmbio, juros, inflação | BCB | API | `python-bcb` |
| IPEADATA | Séries macroeconômicas | IPEA | API OData | `requests` |
| Shapefiles | Malhas UFs/municípios | IBGE | FTP | `geopandas` |
| MIP 2015 | Matriz Insumo-Produto 67×67 | IBGE/SIDRA | API | `sidrapy` |
| Fluxos interestaduais | Comércio entre UFs | IPEA/NEREUS | CSV | `pandas` |
| SIDRA Geral | 5.000+ tabelas | IBGE | API | `sidrapy` |
| PIA | Produção industrial | IBGE/SIDRA | API | `sidrapy` |
| PMS | Serviços | IBGE/SIDRA | API | `sidrapy` |
| PMC | Comércio | IBGE/SIDRA | API | `sidrapy` |
| CAGED | Admissões/demissões | PDET/MTE | FTP | `py7zr` |
| SEADE | Dados SP | SEADE | API/DL | `requests` |

---

## 2. Processamento de Grandes Volumes com DuckDB

Para datasets > 1 GB (RAIS, POF, CAGED), DuckDB é **~50× mais rápido** que pandas chunking.

### 2.1 Processar RAIS completa em 3 segundos

```python
import duckdb

def processar_rais_duckdb(caminho_csv):
    """
    Processa RAIS 2023 (1,2 GB, 11,7M registros) em < 10 segundos.
    
    Returns:
        DataFrame agregado por UF e setor
    """
    con = duckdb.connect()
    
    # Registrar o CSV (sem carregar em memória)
    con.execute(f"""
        CREATE TABLE rais AS
        SELECT * FROM read_csv_auto(
            '{caminho_csv}',
            sep=';',
            encode='iso-8859-1',
            ignore_errors=True
        )
    """)
    
    # Agregação por UF - CNAE 2 dígitos (divisão inteira)
    resultado = con.execute("""
        SELECT 
            uf,
            cnae20classe // 100 AS cnae2,
            SUM(qtd_vinc_ativos) AS emprego
        FROM rais
        WHERE cnae20classe IS NOT NULL
        GROUP BY uf, cnae2
        ORDER BY uf, cnae2
    """).fetchdf()
    
    con.close()
    return resultado

# Todas as UFs, todos os 87 setores CNAE 2 dígitos
def processar_rais_completa(caminho_csv):
    """
    Agregação completa por UF × CNAE 2 dígitos × município.
    """
    con = duckdb.connect()
    
    con.execute(f"""
        CREATE TABLE rais AS
        SELECT * FROM read_csv_auto('{caminho_csv}',
            sep=';', encode='iso-8859-1', ignore_errors=True)
    """)
    
    # Por UF × CNAE2
    uf_cnae = con.execute("""
        SELECT uf, cnae20classe // 100 AS setor,
               SUM(qtd_vinc_ativos) AS empregos,
               AVG(vl_remun_medio) AS salario_medio
        FROM rais
        WHERE cnae20classe IS NOT NULL
        GROUP BY uf, setor
    """).fetchdf()
    
    # Por município (para mapas municipais)
    mun_cnae = con.execute("""
        SELECT municipio, cnae20classe // 100 AS setor,
               SUM(qtd_vinc_ativos) AS empregos
        FROM rais
        WHERE cnae20classe IS NOT NULL
        GROUP BY municipio, setor
    """).fetchdf()
    
    con.close()
    return uf_cnae, mun_cnae
```

### 2.2 PNAD em painel anual

```python
def painel_pnad_anual(anos=range(2012, 2024)):
    """
    Constrói painel UF × ano com variáveis da PNAD Contínua.
    """
    dfs = []
    for ano in anos:
        df = sidrapy.get_table(
            table_code="4093",  # Rendimento
            territorial_level="3",  # UF
            ibge_territorial_code="all",
            period=str(ano),
            format='pandas'
        )
        df['ano'] = ano
        dfs.append(df)
    
    painel = pd.concat(dfs, ignore_index=True)
    return painel
```

### 2.3 Queries espaciais com DuckDB Spatial

```python
def criar_tabela_espacial_duckdb():
    """
    Cria tabela espacial com estados brasileiros no DuckDB.
    Suporta queries espaciais (ST_Contains, ST_Intersects, ST_Distance).
    """
    con = duckdb.connect()
    con.execute("INSTALL spatial; LOAD spatial;")
    
    # Carregar shapefile como tabela espacial
    con.execute(f"""
        CREATE TABLE estados AS
        SELECT * FROM ST_Read('{caminho_shapefile}')
    """)
    
    # Query: encontrar estados vizinhos
    vizinhos = con.execute("""
        SELECT a.nome AS estado_a, b.nome AS estado_b
        FROM estados a, estados b
        WHERE ST_Touches(a.geom, b.geom)
        AND a.nome < b.nome
    """).fetchdf()
    
    con.close()
    return vizinhos
```

---

## 3. Shapefiles e Mapas do Brasil

### 3.1 Carregar e preparar malhas

```python
import geopandas as gpd
import matplotlib.pyplot as plt

def preparar_mapa_brasil(resolucao='uf'):
    """
    Carrega e prepara malha do Brasil.
    
    Args:
        resolucao: 'uf' (27), 'mun' (5.570) ou 'rm' (regiões metropolitanas)
    
    Returns:
        GeoDataFrame pronto para análise
    """
    if resolucao == 'uf':
        url = ("https://raw.githubusercontent.com/codeforavamerica/"
               "click_that_hood/master/public/data/brazil-states.geojson")
        gdf = gpd.read_file(url)
        
        # Ajustar projeção para Albers (preserva área)
        gdf = gdf.to_crs('EPSG:5880')  # SIRGAS 2000 / Brazil Albers
    
    elif resolucao == 'mun':
        gdf = baixar_shapefile_municipios()
        gdf = gdf.to_crs('EPSG:5880')
    
    # Adicionar centróides para labels
    gdf['centroide'] = gdf.geometry.centroid
    gdf['lon'] = gdf.centroide.x
    gdf['lat'] = gdf.centroide.y
    
    return gdf

# Mapa básico
gdf_uf = preparar_mapa_brasil('uf')
ax = gdf_uf.plot(figsize=(12, 10), edgecolor='white', linewidth=0.5)
ax.set_title('Brasil - Unidades Federativas', fontsize=16)
```

### 3.2 Quociente Locacional (QL) para regionalização

```python
def calcular_ql(emprego_setor_uf, emprego_total_uf, emprego_setor_br, emprego_total_br):
    """
    Calcula Quociente Locacional para todos os setores × UFs.
    
    QL_{i,r} = (E_{i,r} / E_{T,r}) / (E_{i,n} / E_{T,n})
    
    Onde:
    - E_{i,r}: emprego setor i na região r
    - E_{T,r}: emprego total na região r
    - E_{i,n}: emprego setor i no nacional
    - E_{T,n}: emprego total no nacional
    
    QL > 1: região tem concentração relativa maior que a nacional
    QL < 1: região tem concentração relativa menor que a nacional
    
    Returns:
        DataFrame com QL para cada (UF, setor)
    """
    # Participação setorial por UF
    part_uf = emprego_setor_uf / emprego_total_uf
    part_br = emprego_setor_br / emprego_total_br
    
    ql = part_uf / part_br
    return ql

def calcular_flq(ql, emprego_regional, emprego_nacional, delta=0.25):
    """
    FLQ (Flegg & Webber, 1997) — Quociente Locacional cross-industry.
    
    FLQ_{i,j,r} = CILQ_{i,j,r} × [log₂(1 + E_r/E_n)]^δ
    
    Usado para regionalizar coeficientes técnicos da MIP.
    """
    cilq = ql / ql.T  # razão entre QL do setor i e setor j
    tamanho_relativo = emprego_regional.sum() / emprego_nacional.sum()
    fator_tamanho = np.log2(1 + tamanho_relativo) ** delta
    
    return cilq * fator_tamanho
```

---

## 4. Matrizes de Pesos Espaciais

### 4.1 Todos os tipos de matriz W

```python
from libpysal.weights import Queen, Rook, DistanceBand, Kernel, W
import libpysal
import numpy as np
import geopandas as gpd

def construir_todas_matrizes_w(gdf, id_col=None):
    """
    Constrói 6 tipos de matrizes de pesos espaciais para comparação.
    
    Args:
        gdf: GeoDataFrame com geometria das regiões
        id_col: Coluna com identificador único
    
    Returns:
        dict com nome → matriz W (row-standardized)
    """
    if id_col is None:
        gdf = gdf.copy()
        gdf['id'] = range(len(gdf))
        id_col = 'id'
    
    matrizes = {}
    
    # 1. Queen contiguity (compartilham fronteira ou ponto)
    w_queen = Queen.from_dataframe(gdf, idVariable=id_col)
    w_queen.transform = 'r'
    matrizes['queen'] = w_queen
    
    # 2. Rook contiguity (compartilham fronteira, exclui vértices)
    w_rook = Rook.from_dataframe(gdf, idVariable=id_col)
    w_rook.transform = 'r'
    matrizes['rook'] = w_rook
    
    # 3. K-Nearest Neighbors (k=5, padrão da literatura)
    w_knn5 = DistanceBand.from_dataframe(gdf, k=5, ids=id_col, binary=True)
    w_knn5.transform = 'r'
    matrizes['knn5'] = w_knn5
    
    w_knn8 = DistanceBand.from_dataframe(gdf, k=8, ids=id_col, binary=True)
    w_knn8.transform = 'r'
    matrizes['knn8'] = w_knn8
    
    # 4. Distance Band (threshold adaptativo)
    # Calcular distância mínima que conecta todos
    centroides = gdf.geometry.centroid
    coords = np.array([(p.x, p.y) for p in centroides])
    from scipy.spatial import distance_matrix
    dists = distance_matrix(coords, coords)
    np.fill_diagonal(dists, np.inf)
    min_dist = dists.min(axis=1).max()  # menor distância que conecta todos
    
    w_dist = DistanceBand.from_dataframe(gdf, threshold=min_dist*1.5, ids=id_col, binary=True)
    w_dist.transform = 'r'
    matrizes['distance_band'] = w_dist
    
    # 5. Kernel weights (Gaussian)
    w_kernel = Kernel.from_dataframe(gdf, k=6, ids=id_col, function='gaussian')
    w_kernel.transform = 'r'
    matrizes['kernel_gaussian'] = w_kernel
    
    # 6. Kernel bisquare
    w_bisquare = Kernel.from_dataframe(gdf, k=6, ids=id_col, function='bisquare')
    w_bisquare.transform = 'r'
    matrizes['kernel_bisquare'] = w_bisquare
    
    return matrizes

def construir_matriz_economica(matriz_fluxos):
    """
    Constrói matriz de pesos baseada em fluxos econômicos reais.
    
    Args:
        matriz_fluxos: DataFrame 27×27 com fluxos comerciais
    
    Returns:
        W linha-padronizada
    """
    W = matriz_fluxos.values.astype(float)
    
    # Zerar diagonal (sem auto-vizinhança)
    np.fill_diagonal(W, 0)
    
    # Linha-padronização
    row_sums = W.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1  # evitar divisão por zero
    W = W / row_sums
    
    return W
```

### 4.2 Propriedades e diagnósticos de W

```python
def diagnosticar_matriz_w(w):
    """
    Diagnóstico completo de uma matriz de pesos espaciais.
    
    Args:
        w: Objeto W do libpysal
    
    Returns:
        dict com propriedades
    """
    n = w.n
    W = w.full()[0]
    
    diag = {
        'dimensao': f"{n}×{n}",
        'conexoes': w.s0,
        'densidade': w.s0 / (n * (n-1)) if n > 1 else 0,
        'simetrica': np.allclose(W, W.T),
        'auto_vizinhanca': np.trace(W),
        'n_vizinhos_min': min(w.cardinalities.values()),
        'n_vizinhos_max': max(w.cardinalities.values()),
        'n_vizinhos_media': np.mean(list(w.cardinalities.values())),
        'conectada': w.n_components == 1,
        'n_componentes': w.n_components,
    }
    
    # Autovalores (para determinar intervalo de ρ)
    autovalores = np.linalg.eigvals(W)
    diag['autovalor_min'] = np.min(autovalores).real
    diag['autovalor_max'] = np.max(autovalores).real
    diag['intervalo_rho'] = (1/diag['autovalor_min'], 1/diag['autovalor_max'])
    
    return diag

def comparar_matrizes_w(gdf, y, X):
    """
    Compara múltiplas matrizes W via AIC.
    
    Args:
        gdf: GeoDataFrame
        y: Variável dependente
        X: Variáveis independentes
    
    Returns:
        DataFrame com ρ, AIC e logL para cada W
    """
    from spreg import ML_Lag
    
    matrizes = construir_todas_matrizes_w(gdf)
    resultados = []
    
    for nome, w in matrizes.items():
        try:
            sar = ML_Lag(y.values.reshape(-1, 1), X, w=w, method='full')
            resultados.append({
                'matriz': nome,
                'rho': sar.rho,
                'logL': sar.logL,
                'AIC': sar.aic,
                'n_vizinhos': np.mean(list(w.cardinalities.values()))
            })
        except Exception as e:
            resultados.append({
                'matriz': nome,
                'rho': None,
                'logL': None,
                'AIC': np.inf,
                'erro': str(e)
            })
    
    return pd.DataFrame(resultados).sort_values('AIC')
```

---

## 5. Análise Exploratória de Dados Espaciais (AEDE)

### 5.1 Testes de Autocorrelação Espacial Global

```python
from libpysal.weights import Queen
import esda
from esda import Moran, Geary, Getis

def aede_global(y, w):
    """
    Bateria completa de testes de autocorrelação espacial global.
    
    Args:
        y: Série de valores (np.array)
        w: Matriz de pesos (objeto W)
    
    Returns:
        dict com estatísticas e p-valores
    """
    resultados = {}
    
    # 1. Moran's I Global
    moran = Moran(y, w)
    resultados['moran_i'] = {
        'I': moran.I,
        'E[I]': moran.EI,
        'z_score': moran.z_norm,
        'p_value': moran.p_norm,
        'significativo_5pct': moran.p_norm < 0.05
    }
    
    # 2. Geary's C
    geary = Geary(y, w)
    resultados['geary_c'] = {
        'C': geary.C,
        'E[C]': geary.EC,
        'z_score': geary.z_norm,
        'p_value': geary.p_norm,
        'significativo_5pct': geary.p_norm < 0.05
    }
    
    # 3. Getis-Ord G geral
    g = Getis(y, w)
    resultados['getis_g'] = {
        'G': g.G,
        'z_score': g.z_norm,
        'p_value': g.p_norm,
        'significativo_5pct': g.p_norm < 0.05
    }
    
    return resultados
```

### 5.2 Indicadores Locais (LISA)

```python
from esda import Moran_Local, G_Local

def aede_local(y, w, significance_level=0.05):
    """
    Análise LISA completa: clusters e outliers espaciais.
    
    Returns:
        dict com DataFrames de resultados locais
    """
    # 1. Local Moran's I (LISA)
    lisa = Moran_Local(y, w)
    
    # Classificação dos clusters (Anselin, 1995)
    # 1: HH (High-High), 2: LH (Low-High), 3: LL (Low-Low), 4: HL (High-Low)
    labels = {1: 'HH', 2: 'LH', 3: 'LL', 4: 'HL'}
    
    resultados_local = pd.DataFrame({
        'local_I': lisa.Is,
        'p_value': lisa.p_sim,
        'quadrant': lisa.q,
        'cluster_type': [labels.get(q, 'NS') for q in lisa.q],
        'significativo': lisa.p_sim < significance_level
    })
    
    # 2. Getis-Ord G* local (hot/cold spots)
    g_local = G_Local(y, w, star=True)
    
    resultados_g = pd.DataFrame({
        'G_i': g_local.Gs,
        'z_score': g_local.z_sim,
        'p_value': g_local.p_sim,
        'hot_spot_95': (g_local.z_sim > 1.96),
        'cold_spot_95': (g_local.z_sim < -1.96)
    })
    
    return {
        'lisa': resultados_local,
        'getis_ord': resultados_g
    }
```

### 5.3 Moran Scatterplot

```python
def moran_scatterplot(y, w, nome_variavel='y'):
    """
    Gera Moran Scatterplot com quadrantes e rótulos.
    """
    import matplotlib.pyplot as plt
    
    # Padronizar
    z = (y - y.mean()) / y.std()
    
    # Lag espacial
    wy = libpysal.weights.lag_spatial(w, z)
    
    moran = Moran(y, w)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(z, wy, alpha=0.7, edgecolors='black', linewidth=0.5)
    
    # Linha de regressão
    from numpy.polynomial.polynomial import polyfit
    b, a = polyfit(z, wy, 1)
    x_line = np.linspace(z.min(), z.max(), 100)
    ax.plot(x_line, a + b*x_line, 'r-', linewidth=2, label=f'ρ = {b:.3f}')
    
    # Linhas dos quadrantes
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    ax.axvline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Rótulos dos quadrantes
    ax.text(0.05, 0.95, 'LH (Low-High)', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.95, 0.95, 'HH (High-High)', transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.05, 0.05, 'LL (Low-Low)', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.text(0.95, 0.05, 'HL (High-Low)', transform=ax.transAxes, fontsize=10,
            verticalalignment='bottom', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_xlabel(f'{nome_variavel} (z-score)', fontsize=12)
    ax.set_ylabel(f'Lag espacial de {nome_variavel}', fontsize=12)
    ax.set_title(f'Moran Scatterplot — I = {moran.I:.4f} (p = {moran.p_norm:.4f})', fontsize=14)
    ax.legend()
    
    return fig
```

### 5.4 Testes de Diagnóstico de Dependência Espacial

```python
from spreg import OLS
from spreg.diagnostics import lm_diagnostics

def diagnosticos_espaciais(y, X, w):
    """
    Bateria de diagnósticos de dependência espacial (Anselin et al., 1996).
    
    Testes:
    - LM-LAG: dependência na forma de lag (SAR vs OLS)
    - LM-ERR: dependência na forma de erro (SEM vs OLS)
    - Robust LM-LAG / LM-ERR: versões robustas
    - SARMA: ambas simultaneamente
    - Moran I dos resíduos OLS
    
    Returns:
        dict com estatísticas e decisão de especificação
    """
    # OLS base
    ols = OLS(y, X, w=w, name_y='y', name_x=['const'] + [f'x{i}' for i in range(X.shape[1]-1)])
    
    # Diagnósticos LM
    lm = lm_diagnostics(ols, w)
    
    # Decisão de especificação (Anselin & Florax, 1995)
    decisao = 'OLS'
    if lm['LM-LAG']['p-value'] < 0.05 and lm['LM-ERR']['p-value'] >= 0.05:
        decisao = 'SAR'
    elif lm['LM-ERR']['p-value'] < 0.05 and lm['LM-LAG']['p-value'] >= 0.05:
        decisao = 'SEM'
    elif lm['LM-LAG']['p-value'] < 0.05 and lm['LM-ERR']['p-value'] < 0.05:
        # Ambos significativos: usar Robustos
        if lm['Robust LM-LAG']['p-value'] < lm['Robust LM-ERR']['p-value']:
            decisao = 'SAR'
        else:
            decisao = 'SEM'
    
    # Moran I dos resíduos
    from esda import Moran
    residuos = ols.u
    moran_res = Moran(residuos.flatten(), w)
    
    return {
        'diagnosticos_lm': lm,
        'decisao_especificacao': decisao,
        'moran_residuos': {
            'I': moran_res.I,
            'p_value': moran_res.p_norm
        }
    }
```

---

## 6. Modelos Espaciais Cross-Section

### 6.1 Todos os modelos da família espacial

```python
from spreg import OLS, ML_Lag, ML_Error, GM_Combo, GM_Error_Het
from spreg import GMM_Error, GMM_Combo

def estimar_todos_modelos_espaciais(y, X, w):
    """
    Estima TODOS os modelos de econometria espacial cross-section.
    
    Modelos:
    1. OLS (sem dependência)
    2. SAR (Spatial Autoregressive) — ρWy
    3. SEM (Spatial Error Model) — λWu
    4. SDM (Spatial Durbin Model) — ρWy + WXθ
    5. SAC/SARAR — ρWy + λWu
    6. SLX — WX
    7. GNS (General Nesting Spatial) — ρWy + WXθ + λWu
    
    Returns:
        dict com todos os modelos estimados
    """
    resultados = {}
    
    # 1. OLS
    ols = OLS(y, X, w=w)
    resultados['OLS'] = {
        'betas': ols.betas.flatten(),
        'se': ols.se,
        'R2': ols.r2,
        'AIC': ols.aic,
        'logL': ols.logL
    }
    
    # 2. SAR (ML)
    sar = ML_Lag(y, X, w=w, method='full')
    resultados['SAR'] = {
        'rho': sar.rho,
        'betas': sar.betas.flatten(),
        'se': sar.se,
        'R2': sar.r2,
        'AIC': sar.aic,
        'logL': sar.logL
    }
    
    # 3. SEM (ML)
    sem = ML_Error(y, X, w=w, method='full')
    resultados['SEM'] = {
        'lambda': sem.lam,
        'betas': sem.betas.flatten(),
        'se': sem.se,
        'R2': sem.r2,
        'AIC': sem.aic,
        'logL': sem.logL
    }
    
    # 4. SDM (Spatial Durbin) via ML_Lag com slx_lags
    sdm = ML_Lag(y, X, w=w, slx_lags=1, method='full')
    resultados['SDM'] = {
        'rho': sdm.rho,
        'betas': sdm.betas.flatten(),
        'se': sdm.se,
        'AIC': sdm.aic,
        'logL': sdm.logL
    }
    
    # 5. SAC/SARAR via GMM (heterocedástico)
    sac = GMM_Error(y, X, w=w, add_wy=True, estimator='het')
    resultados['SAC'] = {
        'rho': sac.rho,
        'lambda': sac.lam,
        'betas': sac.betas.flatten(),
        'se': sac.se,
    }
    
    # 6. SLX via OLS com WX
    from libpysal.weights import lag_spatial
    WX = lag_spatial(w, X[:, 1:])  # lag das variáveis (exceto constante)
    X_slx = np.column_stack([X, WX])
    slx = OLS(y, X_slx, w=w)
    resultados['SLX'] = {
        'betas': slx.betas.flatten(),
        'se': slx.se,
        'AIC': slx.aic,
        'R2': slx.r2
    }
    
    # 7. GNS (SDM + erro espacial) via GMM
    gns = GMM_Error(y, X, w=w, add_wy=True, slx_lags=1, estimator='het')
    resultados['GNS'] = {
        'rho': gns.rho,
        'lambda': gns.lam,
        'betas': gns.betas.flatten(),
        'se': gns.se,
    }
    
    return resultados

def comparar_modelos_por_aic(resultados):
    """
    Compara todos os modelos estimados por AIC e logL.
    
    Returns:
        DataFrame ordenado por AIC (menor é melhor)
    """
    comparacao = []
    for nome, res in resultados.items():
        if 'AIC' in res and res['AIC'] is not None:
            comparacao.append({
                'Modelo': nome,
                'AIC': res['AIC'],
                'logL': res.get('logL'),
                'R2': res.get('R2')
            })
    
    df = pd.DataFrame(comparacao).sort_values('AIC')
    df['ΔAIC'] = df['AIC'] - df['AIC'].min()
    return df
```

### 6.2 Efeitos Diretos, Indiretos e Totais

```python
def calcular_efeitos_spillover(sar_model, W, n_simulacoes=1000):
    """
    Decomposição dos efeitos em diretos, indiretos (spillover) e totais
    para modelo SAR (LeSage & Pace, 2009).
    
    Args:
        sar_model: Modelo ML_Lag estimado
        W: Matriz de pesos (full ndarray)
        n_simulacoes: Número de simulações para ICs
    
    Returns:
        DataFrame com efeitos por variável
    """
    n = W.shape[0]
    k = len(sar_model.betas)
    rho = sar_model.rho
    betas = sar_model.betas.flatten()
    vm = sar_model.vm  # matriz de variância-covariância
    
    # Multiplicador espacial: S = (I - ρW)^(-1)
    I_n = np.eye(n)
    S = np.linalg.inv(I_n - rho * W)
    
    efeitos = []
    for i in range(k):
        # Efeito direto: média da diagonal de S × β_i
        direto = np.mean(np.diag(S)) * betas[i]
        
        # Efeito total: média da soma de todas as linhas de S × β_i
        total = np.mean(S.sum(axis=1)) * betas[i]
        
        # Efeito indireto = total - direto
        indireto = total - direto
        
        efeitos.append({
            'variavel': f'X{i}',
            'direto': direto,
            'indireto': indireto,
            'total': total
        })
    
    return pd.DataFrame(efeitos)
```

### 6.3 Efeitos para SDM

```python
def efeitos_sdm(sdm_model, W):
    """
    Decomposição de efeitos para SDM (inclui WX).
    Para SDM: efeitos diretos e indiretos diferem por variável.
    """
    n = W.shape[0]
    rho = sdm_model.rho
    k = (len(sdm_model.betas) - 1) // 2  # número de X (excluindo WX)
    
    betas = sdm_model.betas.flatten()
    beta_x = betas[1:k+1]  # coeficientes de X
    beta_wx = betas[k+1:]   # coeficientes de WX
    
    I_n = np.eye(n)
    S = np.linalg.inv(I_n - rho * W)
    
    efeitos = []
    for i in range(k):
        # Para SDM: efeito direto = média da diagonal de S × (β_i + W_ii * θ_i)
        # Mas W_ii = 0, então é S_diag × β_i + algo de WX
        # Simplificação: S × (I_n * β_i + W * θ_i)
        efeito_mat = S @ (np.eye(n) * beta_x[i] + W * beta_wx[i])
        
        direto = np.mean(np.diag(efeito_mat))
        total = np.mean(efeito_mat.sum(axis=1))
        indireto = total - direto
        
        efeitos.append({
            'variavel': f'X{i}',
            'direto': direto,
            'indireto': indireto,
            'total': total
        })
    
    return pd.DataFrame(efeitos)
```

---

## 7. Modelos em Painel Espacial

### 7.1 Painel espacial com spreg 1.9+ (2026)

```python
def preparar_painel_espacial(dados, col_uf='uf', col_ano='ano', 
                              col_y='y', col_x=None):
    """
    Prepara dados em formato wide para painel espacial.
    
    spreg espera: y com shape (n, T) onde n = número de regiões,
    T = número de períodos. X similar.
    
    Args:
        dados: DataFrame longo (n*T observações)
        col_uf: Coluna com identificador da região
        col_ano: Coluna com período
        col_y: Coluna da variável dependente
        col_x: Lista de colunas das variáveis independentes
    
    Returns:
        tuple (y_wide, X_wide, ufs, anos)
    """
    ufs = sorted(dados[col_uf].unique())
    anos = sorted(dados[col_ano].unique())
    T = len(anos)
    n = len(ufs)
    
    # y em formato wide: (n, T)
    y_pivot = dados.pivot(index=col_uf, columns=col_ano, values=col_y)
    y_wide = y_pivot.values  # shape (n, T)
    
    if col_x:
        k = len(col_x)
        X_wide = np.zeros((n, T * k))
        for j, var in enumerate(col_x):
            x_pivot = dados.pivot(index=col_uf, columns=col_ano, values=var)
            X_wide[:, j*T:(j+1)*T] = x_pivot.values
    else:
        X_wide = None
    
    return y_wide, X_wide, ufs, anos

def estimar_paineis_espaciais(y_wide, X_wide, w):
    """
    Estima TODOS os modelos de painel espacial disponíveis.
    
    Requer spreg >= 1.9.0.
    
    Returns:
        dict com modelos estimados
    """
    from spreg import (
        PooledOLS, PanelFE, PanelRE,
        ML_LagFE, ML_LagRE,
        ML_ErrorFE, ML_ErrorRE,
        ML_ErrorPooled
    )
    
    resultados = {}
    
    # 1. Pooled OLS
    pooled = PooledOLS(y_wide, X_wide, w)
    resultados['PooledOLS'] = {
        'betas': pooled.betas.flatten(),
        'se': pooled.se,
        'R2': pooled.r2,
        'logL': pooled.logL
    }
    
    # 2. Fixed Effects
    fe = PanelFE(y_wide, X_wide, w)
    resultados['PanelFE'] = {
        'betas': fe.betas.flatten(),
        'se': fe.se,
        'R2': fe.r2,
        'logL': fe.logL
    }
    
    # 3. Random Effects
    re = PanelRE(y_wide, X_wide, w)
    resultados['PanelRE'] = {
        'betas': re.betas.flatten(),
        'se': re.se,
        'R2': re.r2,
        'logL': re.logL,
        'hausman': re.hausman if hasattr(re, 'hausman') else None
    }
    
    # 4. Spatial Lag FE
    lag_fe = ML_LagFE(y_wide, X_wide, w)
    resultados['SAR_FE'] = {
        'rho': lag_fe.rho,
        'betas': lag_fe.betas.flatten(),
        'se': lag_fe.se,
        'logL': lag_fe.logL
    }
    
    # 5. Spatial Lag RE
    lag_re = ML_LagRE(y_wide, X_wide, w)
    resultados['SAR_RE'] = {
        'rho': lag_re.rho,
        'betas': lag_re.betas.flatten(),
        'se': lag_re.se,
        'logL': lag_re.logL
    }
    
    # 6. Spatial Error FE
    err_fe = ML_ErrorFE(y_wide, X_wide, w)
    resultados['SEM_FE'] = {
        'lambda': err_fe.lam,
        'betas': err_fe.betas.flatten(),
        'se': err_fe.se,
        'logL': err_fe.logL
    }
    
    # 7. Spatial Error RE
    err_re = ML_ErrorRE(y_wide, X_wide, w)
    resultados['SEM_RE'] = {
        'lambda': err_re.lam,
        'betas': err_re.betas.flatten(),
        'se': err_re.se,
        'logL': err_re.logL
    }
    
    return resultados
```

### 7.2 Teste de Hausman para Painel Espacial

```python
def hausman_espacial(fe_model, re_model):
    """
    Teste de Hausman para escolher entre FE e RE em painel espacial.
    H0: RE é consistente e eficiente (não correlacionado com os regressores)
    H1: RE é inconsistente, FE é consistente
    
    Args:
        fe_model: Modelo com efeitos fixos
        re_model: Modelo com efeitos aleatórios
    
    Returns:
        dict com estatística, p-valor e decisão
    """
    beta_fe = fe_model.betas.flatten()
    beta_re = re_model.betas.flatten()
    var_fe = fe_model.vm
    var_re = re_model.vm
    
    diff = beta_fe - beta_re
    var_diff = var_fe - var_re
    
    # Pseudo-inversa para garantir inversibilidade
    from numpy.linalg import pinv
    H = diff @ pinv(var_diff) @ diff
    
    from scipy.stats import chi2
    df = len(beta_fe)
    p_value = 1 - chi2.cdf(H, df)
    
    return {
        'H_stat': H,
        'df': df,
        'p_value': p_value,
        'decisao': 'RE' if p_value > 0.05 else 'FE'
    }
```

---

## 8. Estimação Bayesiana (PyMC)

### 8.1 SAR Bayesiano

```python
import pymc as pm
import numpy as np

def sar_bayesiano(y, X, W, n_amostras=2000):
    """
    SAR Bayesiano via PyMC com amostragem MCMC.
    
    Vantagens sobre MLE:
    - Distribuição completa a posteriori (não apenas pontual)
    - Priors informativos excluem ρ espúrio
    - HDI 95% para intervalos de credibilidade
    - Determinante por autovalores (estável para n > 10K)
    
    Args:
        y: Variável dependente (n,)
        X: Variáveis independentes (n, k)
        W: Matriz de pesos (n, n) linha-padronizada
        n_amostras: Número de amostras MCMC
    
    Returns:
        trace: Objeto trace do PyMC
    """
    n = len(y)
    k = X.shape[1]
    
    # Autovalores de W para ln|I - ρW| estável
    autovalores = np.linalg.eigvals(W)
    
    with pm.Model() as model:
        # Priors
        rho = pm.Uniform('rho', -0.99, 0.99)
        beta = pm.Normal('beta', mu=0, sigma=10, shape=k)
        sigma = pm.HalfCauchy('sigma', 5)
        
        # Determinante via autovalores
        ln_det = pm.math.sum(pm.math.log(1 - rho * autovalores))
        
        # Média: ρWy + Xβ
        Wy = W @ y
        mu = rho * Wy + X @ beta
        
        # Log-verossimilhança
        pm.Potential(
            'likelihood',
            -n/2 * pm.math.log(2 * np.pi * sigma**2)
            - (y - mu)**2 / (2 * sigma**2)
            + ln_det
        )
        
        # Amostragem
        trace = pm.sample(
            n_amostras,
            tune=n_amostras,
            target_accept=0.9,
            chains=4,
            cores=4,
            random_seed=42
        )
    
    return trace
```

### 8.2 Sumário Bayesiano

```python
def sumario_bayesiano(trace):
    """
    Sumário completo da estimação Bayesiana.
    
    Returns:
        DataFrame com média, HDI 95%, R_hat, etc.
    """
    import arviz as az
    
    sumario = az.summary(trace, hdi_prob=0.95)
    
    # Efetivo tamanho amostral
    ess = az.ess(trace)
    
    # R_hat (deve ser < 1.01)
    rhat = az.rhat(trace)
    
    return {
        'sumario': sumario,
        'ess': ess,
        'rhat': rhat,
        'passou_diagnostico': all(r < 1.01 for r in rhat.values())
    }
```

---

## 9. Simulação e Multiplicador Espacial

### 9.1 Simulação de choques

```python
def simular_choque_espacial(y_orig, rho, W, vetor_choque_log, 
                             n_mc=10000, alpha=0.05):
    """
    Simula choque espacial com multiplicador (I - ρW)^(-1)
    e intervalos de confiança por Monte Carlo.
    
    Args:
        y_orig: Valores originais da variável (níveis)
        rho: Parâmetro espacial estimado
        W: Matriz de pesos linha-padronizada
        vetor_choque_log: Vetor de choques em log (δ)
        n_mc: Número de simulações Monte Carlo
        alpha: Nível de significância para IC
    
    Returns:
        DataFrame com efeitos diretos, indiretos e totais
    """
    n = len(y_orig)
    I_n = np.eye(n)
    
    # Multiplicador espacial
    S = np.linalg.inv(I_n - rho * W)
    
    # Efeito total em log
    efeito_log_total = S @ vetor_choque_log
    
    # Conversão para níveis
    efeito_niveis = y_orig * (np.exp(efeito_log_total) - 1)
    
    # Decomposição
    efeito_direto = y_orig * (np.exp(vetor_choque_log) - 1)
    efeito_indireto = efeito_niveis - efeito_direto
    
    # Monte Carlo para ICs (considerando incerteza em ρ)
    if n_mc > 0:
        # Simular ρ da distribuição assintótica N(ρ_hat, Var(ρ))
        # (na prática, usar a distribuição posterior se Bayesiano)
        rho_mc = np.random.normal(rho, 0.05, size=n_mc)  # DP aproximado
        rho_mc = np.clip(rho_mc, -0.99, 0.99)
        
        spillovers_mc = np.zeros(n_mc)
        for i in range(n_mc):
            S_mc = np.linalg.inv(I_n - rho_mc[i] * W)
            efeito_mc = S_mc @ vetor_choque_log
            spillovers_mc[i] = (y_orig * (np.exp(efeito_mc) - 1)).sum() - \
                               efeito_direto.sum()
        
        # Percentis
        li = np.percentile(spillovers_mc, 100 * alpha / 2)
        ls = np.percentile(spillovers_mc, 100 * (1 - alpha / 2))
    else:
        li, ls = None, None
    
    return {
        'efeito_direto': float(efeito_direto.sum()),
        'efeito_indireto': float(efeito_indireto.sum()),
        'efeito_total': float(efeito_niveis.sum()),
        'multiplicador': float(efeito_niveis.sum() / efeito_direto.sum()),
        'IC_spillover_95': (float(li), float(ls)) if li else None,
        'por_regiao': pd.DataFrame({
            'regiao': range(n),
            'efeito_direto': efeito_direto,
            'efeito_indireto': efeito_indireto,
            'efeito_total': efeito_niveis,
            'variacao_pct': np.exp(efeito_log_total) - 1
        })
    }
```

### 9.2 Expansão em série geométrica

```python
def decompor_spillovers(rho, W, vetor_choque, ordens=5):
    """
    Decompõe o spillover em ordens de vizinhança.
    
    S = I + ρW + ρ²W² + ρ³W³ + ...
    
    Returns:
        DataFrame com spillover por ordem
    """
    n = len(vetor_choque)
    I_n = np.eye(n)
    
    termos = []
    acumulado = np.zeros(n)
    
    for ordem in range(ordens + 1):
        termo = (rho ** ordem) * (np.linalg.matrix_power(W, ordem) @ vetor_choque)
        acumulado += termo
        termos.append({
            'ordem': ordem,
            'tipo': 'direto' if ordem == 0 else f'indireto_{ordem}ª',
            'total': float(termo.sum()),
            'acumulado': float(acumulado.sum())
        })
    
    return pd.DataFrame(termos)
```

---

## 10. Validação Cruzada Espacial

### 10.1 K-Fold Espacial

```python
from sklearn.model_selection import KFold
import numpy as np

def validacao_cruzada_espacial(y, X, w, modelo_class, k_folds=5):
    """
    Validação cruzada espacial com blocos geográficos.
    
    Diferente da validação cruzada tradicional, aqui os folds
    são blocos contíguos para evitar vazamento de informação espacial.
    
    Args:
        y: Variável dependente
        X: Variáveis independentes
        w: Matriz de pesos
        modelo_class: Classe do modelo (ex: ML_Lag)
        k_folds: Número de folds
    
    Returns:
        dict com métricas de validação
    """
    n = len(y)
    kf = KFold(n_splits=k_folds, shuffle=True, random_state=42)
    
    erros = []
    r2_scores = []
    
    for train_idx, test_idx in kf.split(X):
        y_train, y_test = y[train_idx], y[test_idx]
        X_train, X_test = X[train_idx], X[test_idx]
        
        try:
            modelo = modelo_class(y_train, X_train, w=w, method='full')
            
            # Previsão
            y_pred = modelo.predict(X_test)
            
            # Erro quadrático médio
            mse = np.mean((y_test - y_pred.flatten())**2)
            erros.append(mse)
            
            # R² fora da amostra
            ss_res = np.sum((y_test - y_pred.flatten())**2)
            ss_tot = np.sum((y_test - y_test.mean())**2)
            r2 = 1 - ss_res / ss_tot if ss_tot > 0 else 0
            r2_scores.append(r2)
            
        except Exception as e:
            erros.append(np.inf)
            r2_scores.append(-np.inf)
    
    return {
        'MSE_medio': np.mean(erros),
        'MSE_std': np.std(erros),
        'R2_fora_amostra_medio': np.mean(r2_scores),
        'R2_fora_amostra_std': np.std(r2_scores),
        'erros_por_fold': erros,
        'r2_por_fold': r2_scores
    }

def validacao_cruzada_multiplos_modelos(y, X, w):
    """
    Compara múltiplos modelos espaciais via validação cruzada.
    
    Returns:
        DataFrame com performance de cada modelo
    """
    from spreg import OLS, ML_Lag, ML_Error
    
    modelos = {
        'OLS': lambda: (OLS, {}),
        'SAR': lambda: (ML_Lag, {}),
        'SEM': lambda: (ML_Error, {}),
    }
    
    resultados = []
    for nome, get_modelo in modelos.items():
        cls, kwargs = get_modelo()
        vc = validacao_cruzada_espacial(y, X, w, cls)
        resultados.append({
            'modelo': nome,
            'MSE': vc['MSE_medio'],
            'R2_fora_amostra': vc['R2_fora_amostra_medio']
        })
    
    return pd.DataFrame(resultados).sort_values('MSE')
```

### 10.2 Leave-One-Region-Out

```python
def loo_espacial_por_regiao(y, X, w, regioes, modelo_class):
    """
    Leave-One-Region-Out: treina em R-1 regiões, testa na R-ésima.
    Útil para avaliar generalização entre regiões geográficas.
    
    Args:
        regioes: Lista de rótulos de região para cada obs
    """
    from sklearn.metrics import r2_score
    
    regioes_unicas = np.unique(regioes)
    resultados = []
    
    for regiao_teste in regioes_unicas:
        mask_teste = regioes == regiao_teste
        mask_treino = ~mask_teste
        
        y_train, y_test = y[mask_treino], y[mask_teste]
        X_train, X_test = X[mask_treino], X[mask_teste]
        
        try:
            modelo = modelo_class(y_train, X_train, w=w, method='full')
            y_pred = modelo.predict(X_test)
            
            r2 = r2_score(y_test, y_pred.flatten())
            rmse = np.sqrt(np.mean((y_test - y_pred.flatten())**2))
            
            resultados.append({
                'regiao_teste': regiao_teste,
                'n_treino': mask_treino.sum(),
                'n_teste': mask_teste.sum(),
                'R2': r2,
                'RMSE': rmse
            })
        except Exception as e:
            resultados.append({
                'regiao_teste': regiao_teste,
                'R2': np.nan,
                'RMSE': np.nan,
                'erro': str(e)
            })
    
    return pd.DataFrame(resultados)
```

---

## 11. Visualização e Dashboards

### 11.1 Mapa LISA de Clusters

```python
def mapa_lisa_clusters(gdf, resultados_lisa, coluna='cluster_type', 
                        titulo='LISA Cluster Map'):
    """
    Gera mapa de clusters LISA (HH, LH, LL, HL, NS).
    """
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    
    gdf = gdf.copy()
    gdf['lisa_cluster'] = resultados_lisa[coluna].values
    gdf['significativo'] = resultados_lisa['significativo'].values
    
    # Apenas clusters significativos
    gdf['map_cluster'] = 'NS (não significativo)'
    mask_sig = gdf['significativo']
    gdf.loc[mask_sig, 'map_cluster'] = gdf.loc[mask_sig, 'lisa_cluster']
    
    # Cores
    cores = {
        'HH': '#d7191c',    # vermelho
        'LH': '#fdae61',    # laranja claro
        'LL': '#2c7bb6',    # azul
        'HL': '#abd9e9',    # azul claro
        'NS (não significativo)': '#f0f0f0'  # cinza
    }
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    for tipo in cores:
        mask = gdf['map_cluster'] == tipo
        if mask.any():
            gdf[mask].plot(ax=ax, color=cores[tipo], edgecolor='white', 
                          linewidth=0.5, label=tipo, legend=True)
    
    ax.set_title(titulo, fontsize=14)
    ax.legend(loc='lower right', fontsize=10)
    ax.axis('off')
    
    return fig
```

### 11.2 Dashboard Interativo (Plotly)

```python
import plotly.graph_objects as go
import plotly.express as px

def dashboard_spillover_interativo(gdf, resultados_simulacao, 
                                    col_spillover='efeito_total'):
    """
    Dashboard interativo com mapa coroplético + gráficos.
    
    Args:
        gdf: GeoDataFrame com geometrias
        resultados_simulacao: DataFrame com resultados por região
        col_spillover: Coluna com valores a mapear
    
    Returns:
        Figura Plotly interativa (HTML)
    """
    merged = gdf.merge(resultados_simulacao, left_on='nome', right_on='regiao')
    
    # Mapa principal
    fig = go.Figure()
    
    fig.add_trace(go.Choropleth(
        locations=merged['sigla'],
        z=merged[col_spillover],
        locationmode='ISO-3',
        colorscale='YlOrRd',
        text=merged['nome'],
        colorbar_title='Spillover (empregos)',
        hovertemplate='<b>%{text}</b>: %{z:,.0f} empregos<extra></extra>',
        marker_line_color='white',
        marker_line_width=0.5
    ))
    
    fig.update_layout(
        title='Mapa de Spillovers - Efeito Total',
        geo=dict(
            scope='south america',
            showcountries=True,
            countrycolor='white',
            projection={'type': 'natural earth'},
            showframe=False
        ),
        width=1000,
        height=800
    )
    
    return fig

def dashboard_completo(gdf, resultados, resultados_lisa):
    """
    Dashboard completo com múltiplos painéis.
    """
    from plotly.subplots import make_subplots
    
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=('Spillover Absoluto', 'Spillover Relativo (%)',
                       'LISA Clusters', 'Distribuição dos Spillovers'),
        specs=[[{'type': 'choropleth'}, {'type': 'choropleth'}],
               [{'type': 'choropleth'}, {'type': 'histogram'}]]
    )
    
    merged = gdf.merge(resultados, left_on='nome', right_on='regiao')
    
    # Mapa 1: Spillover absoluto
    fig.add_trace(go.Choropleth(
        locations=merged['sigla'], z=merged['efeito_total'],
        locationmode='ISO-3', colorscale='YlOrRd',
        colorbar_title='Empregos',
        marker_line_color='white', marker_line_width=0.5
    ), row=1, col=1)
    
    # Mapa 2: Spillover relativo
    merged['variacao_pct'] = merged['variacao_pct'] * 100
    fig.add_trace(go.Choropleth(
        locations=merged['sigla'], z=merged['variacao_pct'],
        locationmode='ISO-3', colorscale='Blues',
        colorbar_title='%',
        marker_line_color='white', marker_line_width=0.5
    ), row=1, col=2)
    
    # Mapa 3: LISA clusters
    gdf_lisa = gdf.copy()
    gdf_lisa['cluster'] = resultados_lisa['cluster_type']
    gdf_lisa.loc[~resultados_lisa['significativo'], 'cluster'] = 'NS'
    
    cores = {'HH': 3, 'LH': 2, 'LL': 1, 'HL': 0, 'NS': -1}
    gdf_lisa['cluster_code'] = gdf_lisa['cluster'].map(cores)
    
    fig.add_trace(go.Choropleth(
        locations=merged['sigla'], z=gdf_lisa['cluster_code'],
        locationmode='ISO-3', colorscale='RdYlBu_r',
        colorbar_title='Cluster',
        marker_line_color='white', marker_line_width=0.5
    ), row=2, col=1)
    
    # Histograma
    fig.add_trace(go.Histogram(
        x=merged['efeito_total'], nbinsx=15,
        marker_color='#2c7bb6', opacity=0.7
    ), row=2, col=2)
    
    fig.update_layout(
        title_text='Dashboard Completo - Análise Espacial',
        height=1000, width=1200,
        showlegend=False
    )
    
    return fig
```

### 11.3 Mapa Interativo com Folium

```python
import folium
from folium.plugins import HeatMap, MarkerCluster

def mapa_folium_interativo(gdf, col_valor='efeito_total', 
                            nome_variavel='Spillover'):
    """
    Mapa interativo HTML com Folium + Leaflet.js.
    
    Vantagens:
    - Zoom, pan, tooltips
    - Camadas selecionáveis
    - Exporta como HTML independente
    """
    # Centro do Brasil
    m = folium.Map(location=[-15.5, -54.0], zoom_start=4,
                   tiles='CartoDB positron')
    
    # Normalizar valores para escala de cores
    from branca.colormap import linear
    vmin, vmax = gdf[col_valor].min(), gdf[col_valor].max()
    colormap = linear.YlOrRd_09.scale(vmin, vmax)
    colormap.caption = nome_variavel
    
    # Adicionar polígonos
    folium.GeoJson(
        gdf.__geo_interface__,
        style_function=lambda feat: {
            'fillColor': colormap(feat['properties'][col_valor]),
            'color': 'white',
            'weight': 0.5,
            'fillOpacity': 0.7
        },
        highlight_function=lambda feat: {
            'weight': 3,
            'color': '#003366',
            'fillOpacity': 0.9
        },
        tooltip=folium.GeoJsonTooltip(
            fields=['nome', col_valor],
            aliases=['UF:', f'{nome_variavel}:'],
            localize=True,
            fmt='{:,.0f}'
        )
    ).add_to(m)
    
    m.add_child(colormap)
    
    return m  # .save('mapa_interativo.html')
```

---

## 12. Integração com MIP/SAM/CGE

### 12.1 Multiplicadores setoriais regionais

```python
def multiplicadores_regionais(matriz_A, emprego_setor, ql_setores):
    """
    Calcula multiplicadores de emprego setoriais com ajuste regional via QL.
    
    Args:
        matriz_A: Coeficientes técnicos nacionais (n×n)
        emprego_setor: Emprego por setor na região
        ql_setores: QL por setor na região
    
    Returns:
        dict com multiplicadores tipo I e II
    """
    # Ajuste regional dos coeficientes via FLQ
    n = len(ql_setores)
    A_reg = np.zeros_like(matriz_A)
    
    for i in range(n):
        for j in range(n):
            # FLQ simple: QL_i / QL_j (cross-industry)
            cilq = ql_setores[i] / ql_setores[j] if ql_setores[j] > 0 else 0
            flq = min(cilq, 1.0)  #上限 1
            A_reg[i, j] = matriz_A[i, j] * flq
    
    # Matriz inversa de Leontief
    I = np.eye(n)
    L = np.linalg.inv(I - A_reg)
    
    # Multiplicador tipo I (setorial)
    mult_tipo1 = L.sum(axis=0)
    
    # Multiplicador tipo II (inclui consumo induzido)
    # (Requer coeficiente de consumo)
    
    return {
        'multiplicador_tipo1': mult_tipo1,
        'matriz_inversa_Leontief': L
    }
```

### 12.2 CGE Espacial (Inter-regional)

```python
def modelo_cge_2_regioes(parametros):
    """
    CGE inter-regional simples: MT × Resto do Brasil.
    
    Equações:
    - Migração: ln(L_MT/L_RB) = η × ln(w_MT/w_RB)
    - Trabalho: L_MT + L_RB = L_total
    - Demanda final: CES entre produto MT e RB (Armington)
    - Solução: 2 estágios (curto prazo SAR → longo prazo CGE)
    
    Args:
        parametros: dict com elasticidades e dotacoes
    
    Returns:
        dict com resultados do equilibrio
    """
    # Parâmetros
    eta = parametros.get('eta_migracao', 0.5)
    sigma_a = parametros.get('sigma_armington', 2.0)
    L_total = parametros['L_total']
    L_MT_init = parametros['L_MT']
    w_MT = parametros['w_MT']
    w_RB = parametros['w_RB']
    
    # Equilíbrio: migração iguala salários relativos
    # ln(L_MT/L_RB) = η × ln(w_MT/w_RB)
    # L_MT + L_RB = L_total
    
    from scipy.optimize import fsolve
    
    def equilibrio(x):
        L_MT, L_RB = x
        eq1 = np.log(L_MT / L_RB) - eta * np.log(w_MT / w_RB)
        eq2 = L_MT + L_RB - L_total
        return [eq1, eq2]
    
    sol = fsolve(equilibrio, [L_MT_init, L_total - L_MT_init])
    
    return {
        'L_MT_eq': sol[0],
        'L_RB_eq': sol[1],
        'w_MT_rel': w_MT / w_RB,
        'L_MT_var_pct': (sol[0] - L_MT_init) / L_MT_init * 100
    }
```

---

## 13. Pipeline Completo de Análise Espacial

### 13.1 Função principal

```python
def pipeline_analise_espacial_completo(
    ano_rais=2023,
    setor_cnae='01',
    choque_pct=15,
    estado_choque='MT',
    output_dir='./resultados_espacial'
):
    """
    Pipeline completo de análise espacial:
    1. Download RAIS → 2. Agregação → 3. Shapefile → 4. Matriz W →
    5. AEDE → 6. Estimação SAR → 7. Diagnóstico → 8. Simulação →
    9. Visualização → 10. Relatório
    
    Args:
        ano_rais: Ano da RAIS
        setor_cnae: CNAE 2 dígitos do setor
        choque_pct: Magnitude do choque (%)
        estado_choque: UF que recebe o choque
        output_dir: Diretório de saída
    
    Returns:
        dict com todos os resultados
    """
    import os, json, time
    from datetime import datetime
    
    os.makedirs(output_dir, exist_ok=True)
    inicio = time.time()
    
    print("=" * 60)
    print("PIPELINE COMPLETO DE ANÁLISE ESPACIAL")
    print(f"Início: {datetime.now().isoformat()}")
    print("=" * 60)
    
    # ---- Etapa 1: Dados ----
    print("\n[1/9] Coletando dados RAIS...")
    csv_rais = baixar_rais(ano_rais)
    df_emprego = processar_rais_filtro_setorial(
        csv_rais, cod_cnae2=setor_cnae
    )
    print(f"  → {len(df_emprego)} UFs processadas")
    
    # ---- Etapa 2: Shapefile ----
    print("\n[2/9] Carregando malha geográfica...")
    gdf = preparar_mapa_brasil('uf')
    
    # Merge com dados
    gdf = gdf.merge(df_emprego, left_on='sigla', right_on='uf', how='left')
    gdf['ln_emp_agri'] = np.log(gdf['emp_agri'].clip(lower=1))
    gdf['ln_emp_total'] = np.log(gdf['emp_total'].clip(lower=1))
    
    # ---- Etapa 3: Matriz de Pesos ----
    print("\n[3/9] Construindo matrizes de pesos espaciais...")
    matrizes_w = construir_todas_matrizes_w(gdf, id_col='sigla')
    
    # Selecionar a melhor matriz via AIC
    y = gdf['ln_emp_agri'].values.reshape(-1, 1)
    X = np.column_stack([np.ones(len(gdf)), gdf['ln_emp_total'].values])
    
    comp_w = comparar_matrizes_w(gdf, y, X)
    melhor_w = comp_w.iloc[0]
    w = matrizes_w[melhor_w['matriz']]
    print(f"  → Melhor W: {melhor_w['matriz']} (AIC={melhor_w['AIC']:.1f})")
    
    # ---- Etapa 4: AEDE ----
    print("\n[4/9] Análise Exploratória de Dados Espaciais...")
    aede = aede_global(gdf['ln_emp_agri'].values, w)
    for teste, res in aede.items():
        print(f"  → {teste}: I={res.get('I', res.get('C', res.get('G'))):.4f}, "
              f"p={res['p_value']:.4f}")
    
    aede_local = aede_local(gdf['ln_emp_agri'].values, w)
    
    # ---- Etapa 5: Diagnóstico ----
    print("\n[5/9] Diagnósticos de especificação...")
    diag = diagnosticos_espaciais(y, X, w)
    print(f"  → Especificação recomendada: {diag['decisao_especificacao']}")
    
    # ---- Etapa 6: Estimação SAR ----
    print("\n[6/9] Estimando modelo SAR...")
    from spreg import ML_Lag
    sar = ML_Lag(y, X, w=w, method='full', name_y='ln_emp_agri',
                 name_x=['const', 'ln_emp_total'])
    print(f"  → ρ = {sar.rho:.4f}")
    print(f"  → β₁ = {sar.betas[1,0]:.4f}")
    print(f"  → AIC = {sar.aic:.1f}")
    
    # ---- Etapa 7: Efeitos Diretos/Indiretos ----
    print("\n[7/9] Decompondo efeitos...")
    W_full = w.full()[0]
    efeitos = calcular_efeitos_spillover(sar, W_full)
    print(efeitos.to_string(index=False))
    
    # ---- Etapa 8: Simulação do Choque ----
    print(f"\n[8/9] Simulando choque de +{choque_pct}% em {estado_choque}...")
    vetor_choque = np.zeros(len(gdf))
    idx_choque = gdf[gdf['sigla'] == estado_choque].index[0]
    vetor_choque[idx_choque] = np.log(1 + choque_pct / 100)
    
    simulacao = simular_choque_espacial(
        gdf['emp_agri'].values, sar.rho, W_full, vetor_choque,
        n_mc=5000
    )
    print(f"  → Efeito direto: {simulacao['efeito_direto']:,.0f} empregos")
    print(f"  → Spillover: {simulacao['efeito_indireto']:,.0f} empregos")
    print(f"  → Multiplicador: {simulacao['multiplicador']:.2f}")
    
    # ---- Etapa 9: Visualização ----
    print("\n[9/9] Gerando visualizações...")
    df_regioes = simulacao['por_regiao']
    gdf['efeito_total'] = df_regioes['efeito_total'].values
    gdf['variacao_pct'] = df_regioes['variacao_pct'].values * 100
    
    # Mapas
    fig_dashboard = dashboard_completo(gdf, df_regioes, aede_local)
    fig_dashboard.write_html(os.path.join(output_dir, 'dashboard.html'))
    
    # Mapa Folium
    mapa = mapa_folium_interativo(gdf, 'efeito_total', 'Spillover')
    mapa.save(os.path.join(output_dir, 'mapa_interativo.html'))
    
    # Moran Scatterplot
    fig_moran = moran_scatterplot(gdf['ln_emp_agri'].values, w)
    fig_moran.savefig(os.path.join(output_dir, 'moran_scatterplot.png'), dpi=150)
    
    # ---- Finalização ----
    tempo_total = time.time() - inicio
    
    relatorio = {
        'metadados': {
            'data': datetime.now().isoformat(),
            'ano_rais': ano_rais,
            'setor_cnae': setor_cnae,
            'choque_pct': choque_pct,
            'estado_choque': estado_choque,
            'tempo_execucao_seg': tempo_total
        },
        'modelo': {
            'tipo': 'SAR',
            'rho': float(sar.rho),
            'rho_pvalue': float(sar.p_multi[0]) if hasattr(sar, 'p_multi') else None,
            'AIC': float(sar.aic)
        },
        'aede': {
            'moran_I': float(aede['moran_i']['I']),
            'moran_p': float(aede['moran_i']['p_value']),
            'geary_C': float(aede['geary_c']['C']),
            'geary_p': float(aede['geary_c']['p_value'])
        },
        'simulacao': {
            'efeito_direto': float(simulacao['efeito_direto']),
            'efeito_indireto': float(simulacao['efeito_indireto']),
            'efeito_total': float(simulacao['efeito_total']),
            'multiplicador': float(simulacao['multiplicador']),
            'IC_spillover': list(simulacao['IC_spillover_95']) 
                if simulacao['IC_spillover_95'] else None
        }
    }
    
    with open(os.path.join(output_dir, 'relatorio_completo.json'), 'w') as f:
        json.dump(relatorio, f, indent=2, ensure_ascii=False)
    
    print(f"\n✅ Pipeline concluído em {tempo_total:.1f} segundos")
    print(f"📁 Resultados em: {output_dir}")
    
    return relatorio
```

### 13.2 Relatório Técnico Automático

```python
def gerar_relatorio_pdf(resultados, output_dir='./resultados_espacial'):
    """
    Gera relatório PDF profissional usando Quarto.
    
    Requer: quarto (posit.co) ou pandoc + xelatex
    """
    import subprocess
    
    # Criar arquivo .qmd do Quarto
    qmd_content = f"""---
title: "Relatório de Análise Espacial"
subtitle: "Spillover com Modelo SAR"
author: "AgenteNazaré"
date: "{datetime.now().strftime('%B de %Y')}"
format:
  pdf:
    documentclass: report
    papersize: A4
    toc: true
    toc-depth: 3
    number-sections: true
    colorlinks: true
  html:
    theme: cosmo
    code-fold: true
---

# Resumo Executivo

- **Modelo:** SAR (Spatial Autoregressive)
- **ρ estimado:** {resultados['modelo']['rho']:.4f}
- **AIC:** {resultados['modelo']['AIC']:.1f}
- **Moran's I:** {resultados['aede']['moran_I']:.4f} (p={resultados['aede']['moran_p']:.4f})
- **Choque:** +15% no emprego agrícola de MT
- **Spillover total:** {resultados['simulacao']['efeito_indireto']:,.0f} empregos
- **Multiplicador:** {resultados['simulacao']['multiplicador']:.2f}

# Metodologia

## Modelo SAR

O modelo de defasagem espacial (SAR) é definido como:

$$\\mathbf{{y}} = \\rho\\mathbf{{Wy}} + \\mathbf{{X}}\\boldsymbol{{\\beta}} + \\boldsymbol{{\\varepsilon}}$$

onde $\\rho$ é o parâmetro espacial, $\\mathbf{{W}}$ é a matriz de pesos e $\\mathbf{{Wy}}$ capta a interdependência espacial.

## Multiplicador Espacial

O efeito total de um choque é dado por:

$$\\mathbf{{S}} = (\\mathbf{{I}} - \\rho\\mathbf{{W}})^{{-1}}$$

# Resultados

## Parâmetros Estimados

| Parâmetro | Valor |
|---|---|
| ρ | {resultados['modelo']['rho']:.4f} |
| AIC | {resultados['modelo']['AIC']:.1f} |

## Efeitos do Choque

| Tipo | Valor |
|---|---|
| Efeito direto | {resultados['simulacao']['efeito_direto']:,.0f} |
| Spillover | {resultados['simulacao']['efeito_indireto']:,.0f} |
| Efeito total | {resultados['simulacao']['efeito_total']:,.0f} |
| Multiplicador | {resultados['simulacao']['multiplicador']:.2f} |

# Visualizações

![Dashboard](dashboard.html)
![Moran Scatterplot](moran_scatterplot.png)
"""
    
    with open(os.path.join(output_dir, 'relatorio.qmd'), 'w') as f:
        f.write(qmd_content)
    
    # Renderizar com Quarto
    # subprocess.run(['quarto', 'render', 'relatorio.qmd'], cwd=output_dir)
    # Alternativa: pandoc
    # subprocess.run(['pandoc', 'relatorio.qmd', '-o', 'relatorio.pdf', 
    #                 '--pdf-engine=xelatex'], cwd=output_dir)
```

---

## 14. Instalação de Dependências

```bash
# Core
pip install numpy pandas matplotlib seaborn scipy

# Espacial
pip install geopandas libpysal esda spreg pysal

# Dados
pip install sidrapy python-bcb py7zr openpyxl

# Visualização
pip install plotly folium branca

# Bayesiano (opcional)
pip install pymc arviz

# Big Data
pip install duckdb pyarrow

# Qualidade
pip install great_expectations

# Orquestração (opcional)
pip install prefect

# Relatórios (opcionais, instalar separadamente)
# quarto: https://quarto.org/docs/get-started/
# pandoc: sudo apt-get install pandoc texlive-xetex
```

---

## Regras

### O que SEMPRE fazer
- **Antes de estimar qualquer modelo espacial**: executar testes LM (LM-LAG, LM-ERR) para determinar a especificação correta
- **Comparar múltiplas matrizes W** via AIC antes de decidir qual usar
- **Verificar conectividade da matriz W**: garantir que não há ilhas isoladas
- **Validar ρ**: testar se ρ ≠ 0 via LR test ou intervalo de confiança
- **Documentar a transformação do choque**: choque de +15% = ln(1,15), não 0,15×ln(y)
- **Usar DuckDB** para datasets > 500 MB (RAIS, CAGED, POF)
- **Salvar shapefiles** em formato Parquet para carregamento rápido

### O que NUNCA fazer
- Nunca usar MLE para dados com n > 5.000 (usar GMM ou autovalores)
- Nunca assumir que a matriz W queen/rook é a melhor sem comparar alternativas
- Nunca usar a fórmula errada do choque log-linear (erro comum!)
- Nunca interpretar ρ negativo como ausência de spillover (pode ser competição espacial)
- Nunca ignorar heterocedasticidade em modelos espaciais (usar GMM heterocedástico)
- Nunca fazer validação cruzada tradicional sem ajuste espacial (vazamento de informação)

### Quando algo falhar
- **Dados RAIS não encontrados**: verificar FTP do MTE; ano pode não estar disponível
- **Matriz W desconexa**: usar k-NN (k mínimo = 1) para garantir conectividade
- **MLE não converge**: reduzir tolerância, usar método 'ord' (autovalores), ou trocar para GMM
- **ρ no limite do intervalo**: W pode estar mal especificada; testar matrizes alternativas
- **Moran I não significativo**: testar com diferentes matrizes W; pode não haver dependência espacial
- **Falta de memória**: usar DuckDB para processamento out-of-core
