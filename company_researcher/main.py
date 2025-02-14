from typing import TypedDict, List, Annotated, Literal, Dict, Union, Optional 
from datetime import datetime
from tavily import AsyncTavilyClient
import json
import asyncio
import xml.etree.ElementTree as ET
from langchain_openai.chat_models.base import BaseChatOpenAI
from langchain_core.messages import AnyMessage, AIMessage, SystemMessage, ToolMessage
from pydantic import BaseModel, Field
from langchain_core.tools import tool
from langgraph.graph import StateGraph, START, END, add_messages
import os
from dotenv import load_dotenv

load_dotenv()

# Define the research state
class ResearchState(TypedDict):
    company_name: str
    sanctions_data: Dict
    web_research: Dict
    risk_analysis: Dict
    messages: Annotated[list[AnyMessage], add_messages]

class SanctionsMatch(BaseModel):
    """Represents a match in the sanctions dataset"""
    entity_name: str = Field(..., description="Name of the sanctioned entity")
    match_score: float = Field(..., description="Similarity score between company and sanctioned entity")
    sanctions_details: Dict = Field(..., description="Details from the sanctions dataset")

class CompanyRiskAnalysis(BaseModel):
    """Detailed risk analysis for a company"""
    company_name: str = Field(..., description="Name of the company being analyzed")
    sanctions_matches: List[SanctionsMatch] = Field(default_factory=list, description="List of potential sanctions matches")
    risk_level: str = Field(..., description="Overall risk level: LOW, MEDIUM, HIGH, CRITICAL")
    risk_factors: List[str] = Field(..., description="List of identified risk factors")
    recommendations: List[str] = Field(..., description="Recommendations for further due diligence")

class TavilyQuery(BaseModel):
    query: str = Field(description="sub query")
    topic: str = Field(description="type of search, should be 'general' or 'news'")
    days: int = Field(description="number of days back to run 'news' search")
    raw_content: bool = Field(description="include raw content from found sources")
    include_domains: List[str] = Field(
        default=[
            "https://www.suin-juriscol.gov.co/",
            "https://www.riskglobalconsulting.com/",
            "https://www.supersociedades.gov.co/",
            "https://www.uiaf.gov.co/",
            "https://www.treasury.gov/"
        ],
        description="list of domains to include in the research"
    )

class TavilySearchInput(BaseModel):
    sub_queries: List[TavilyQuery] = Field(description="set of sub-queries that can be answered in isolation")

# Función para normalizar nombres
def normalize_name(name):
    """Normaliza un nombre para comparación"""
    return (name.lower()
            .replace(",", "")
            .replace(".", "")
            .replace("-", " ")
            .replace("_", " ")
            .replace("  ", " ")
            .strip())

# Función para comparar nombres
def names_match(name1, name2):
    """
    Compara dos nombres con lógica flexible para manejar nombres parciales.
    Retorna (bool, float) donde bool indica si hay coincidencia y float es el score.
    """
    name1 = normalize_name(name1)
    name2 = normalize_name(name2)
    
    # Dividir los nombres en partes
    parts1 = set(name1.split())
    parts2 = set(name2.split())
    
    # Si alguno de los nombres tiene una sola palabra, ser más flexible
    if len(parts1) == 1 or len(parts2) == 1:
        # Si una palabra está completamente contenida en la otra
        for part in parts1:
            for other_part in parts2:
                if part in other_part or other_part in part:
                    return True
        return False
    
    # Calcular la intersección y unión de palabras
    common_parts = parts1.intersection(parts2)
    all_parts = parts1.union(parts2)
    
    # Calcular similitud de Jaccard
    similarity = len(common_parts) / len(all_parts) if all_parts else 0
    
    # Criterios de coincidencia:
    # 1. Al menos 2 palabras en común
    # 2. O una similitud de Jaccard alta (>0.5) si hay menos palabras
    if len(common_parts) >= 2 or similarity > 0.5:
        return True
    
    # Verificar si las palabras son substrings una de otra
    for part1 in parts1:
        for part2 in parts2:
            if len(part1) >= 4 and len(part2) >= 4:  # Solo para palabras suficientemente largas
                if part1 in part2 or part2 in part1:
                    return True
    
    return False

@tool("analyze_sanctions_data", return_direct=True)
async def analyze_sanctions_data(company_name: str) -> Dict:
    """
    Analyzes a company against the sanctions dataset.
    
    Args:
        company_name (str): Name of the company to analyze
        
    Returns:
        Dict: Analysis results including potential matches and risk factors
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        xml_path = os.path.join(current_dir, 'CustomizeSanctionsDataset.xml')
        print(f"Buscando archivo XML en: {xml_path}")
        
        if not os.path.exists(xml_path):
            raise FileNotFoundError(f"No se encontró el archivo XML en: {xml_path}")
        
        file_size = os.path.getsize(xml_path)
        print(f"Tamaño del archivo XML: {file_size / (1024*1024):.2f} MB")
        
        print("Iniciando parseo del archivo XML...")
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Obtener el namespace
        default_ns = root.tag.split('}')[0].strip('{') if '}' in root.tag else ''
        ns = {'ns': default_ns} if default_ns else {}
        
        print(f"Buscando entidades...")
        entities = root.findall(f'.//{{{default_ns}}}entities/{{{default_ns}}}entity', {})
        if not entities:
            print("No se encontraron entidades, intentando patrón alternativo...")
            entities = root.findall('.//entity', {})
        
        if not entities:
            raise ValueError("No se encontraron entidades en el archivo XML")
        
        print(f"Encontradas {len(entities)} entidades")
        print(f"Analizando coincidencias para: {company_name}")
        matches = []
        
        for entity in entities:
            try:
                names_element = entity.find(f'.//{{{default_ns}}}names', {})
                if names_element is None:
                    continue
                
                entity_names = []
                
                # Procesar cada entrada de nombre
                for name_entry in names_element.findall(f'.//{{{default_ns}}}name', {}):
                    translations = name_entry.find(f'.//{{{default_ns}}}translations', {})
                    if translations is None:
                        continue
                    
                    # Procesar cada traducción
                    for translation in translations.findall(f'.//{{{default_ns}}}translation', {}):
                        # Obtener el nombre completo formateado
                        formatted_name = translation.find(f'.//{{{default_ns}}}formattedFullName', {})
                        if formatted_name is not None and formatted_name.text:
                            entity_names.append(formatted_name.text)
                        
                        # También obtener partes individuales del nombre
                        name_parts = translation.find(f'.//{{{default_ns}}}nameParts', {})
                        if name_parts is not None:
                            for part in name_parts.findall(f'.//{{{default_ns}}}namePart', {}):
                                value = part.find(f'.//{{{default_ns}}}value', {})
                                if value is not None and value.text:
                                    entity_names.append(value.text)
                
                # Eliminar duplicados y nombres vacíos
                entity_names = list(set(filter(None, entity_names)))
                
                # Buscar coincidencias
                for name in entity_names:
                    if names_match(name, company_name):
                        print(f"\n¡Coincidencia encontrada!")
                        print(f"Nombre en lista: {name}")
                        print(f"Nombres asociados: {entity_names}")
                        
                        # Recopilar detalles
                        details = {
                            "entity_name": name,
                            "all_names": entity_names,
                            "match_score": 1.0,
                            "sanctions_details": {
                                "programs": [],
                                "addresses": [],
                                "id_numbers": [],
                                "additional_info": {}
                            }
                        }
                        
                        # Obtener programas
                        programs = entity.find(f'.//{{{default_ns}}}sanctionsPrograms', {})
                        if programs is not None:
                            for program in programs.findall(f'.//{{{default_ns}}}program', {}):
                                value = program.find(f'.//{{{default_ns}}}value', {})
                                if value is not None and value.text:
                                    details["sanctions_details"]["programs"].append(value.text)
                        
                        # Obtener direcciones
                        addresses = entity.find(f'.//{{{default_ns}}}addresses', {})
                        if addresses is not None:
                            for address in addresses.findall(f'.//{{{default_ns}}}address', {}):
                                value = address.find(f'.//{{{default_ns}}}value', {})
                                if value is not None and value.text:
                                    details["sanctions_details"]["addresses"].append(value.text)
                        
                        # Obtener documentos de identidad
                        documents = entity.find(f'.//{{{default_ns}}}identityDocuments', {})
                        if documents is not None:
                            for doc in documents.findall(f'.//{{{default_ns}}}identityDocument', {}):
                                value = doc.find(f'.//{{{default_ns}}}value', {})
                                if value is not None and value.text:
                                    details["sanctions_details"]["id_numbers"].append(value.text)
                        
                        # Obtener información general
                        general_info = entity.find(f'.//{{{default_ns}}}generalInfo', {})
                        if general_info is not None:
                            details["sanctions_details"]["additional_info"] = {
                                "type": general_info.findtext(f'.//{{{default_ns}}}type', ""),
                                "category": general_info.findtext(f'.//{{{default_ns}}}category', ""),
                                "status": general_info.findtext(f'.//{{{default_ns}}}status', "")
                            }
                        
                        matches.append(details)
                        break  # Salir después de encontrar una coincidencia para esta entidad
            
            except Exception as e:
                print(f"Error procesando entidad: {e}")
                continue
        
        print(f"\nAnálisis completado. Se encontraron {len(matches)} coincidencias")
        return {
            "matches": matches,
            "total_matches": len(matches),
            "total_entries_analyzed": len(entities)
        }
        
    except Exception as e:
        print(f"Error en analyze_sanctions_data: {e}")
        import traceback
        print(f"Traceback completo: {traceback.format_exc()}")
        return {"error": str(e)}

@tool("web_research", args_schema=TavilySearchInput, return_direct=True)
async def web_research(sub_queries: List[TavilyQuery]) -> List[Dict]:
    """
    Realiza búsquedas web utilizando el servicio Tavily.

    Esta función toma una lista de consultas y realiza búsquedas web para cada una,
    utilizando los parámetros especificados en cada consulta. Los resultados de todas
    las búsquedas se combinan y se devuelven.

    Args:
        sub_queries (List[TavilyQuery]): Una lista de objetos TavilyQuery, cada uno
        especificando los parámetros para una búsqueda individual.

    Returns:
        List[Dict]: Una lista de resultados de búsqueda combinados de todas las consultas.
    """
    search_results = []
    for query in sub_queries:
        print(f"Realizando búsqueda web para query: {query.query}")
        try:
            response = await AsyncTavilyClient().search(
                query=query.query,
                topic=query.topic,
                days=query.days,
                include_raw_content=query.raw_content,
                max_results=3,  # Reducido a 3 resultados máximo
                include_domains=query.include_domains
            )
            if 'results' in response:
                search_results.extend(response['results'])
                print(f"- Encontrados {len(response['results'])} resultados para: {query.query}")
                # Mostrar títulos de resultados encontrados
                for result in response['results']:
                    print(f"  * {result.get('title', 'Sin título')}")
            else:
                print(f"- No se encontraron resultados para: {query.query}")
        except Exception as e:
            print(f"Error en la búsqueda '{query.query}': {str(e)}")
            continue
    
    print(f"\nBúsqueda web completada. Total de resultados: {len(search_results)}")
    return search_results

tools = [analyze_sanctions_data, web_research]
tools_by_name = {tool.name: tool for tool in tools}
model = BaseChatOpenAI(
    model='deepseek-chat',
    openai_api_key=os.getenv('DEEPSEEK_API_KEY'),
    openai_api_base='https://api.deepseek.com',
    max_tokens=8000,
    temperature=0
).bind_tools(tools)

async def tool_node(state: ResearchState):
    """Process tool calls and update state"""
    docs = state.get('documents', {})
    docs_str = ""
    msgs = []
    
    for tool_call in state["messages"][-1].tool_calls:
        tool = tools_by_name[tool_call["name"]]
        print(f"\n- Ejecutando {tool_call['name']}...")
        
        try:
            result = await tool.ainvoke(tool_call["args"])
            
            if tool_call["name"] == "web_research":
                # Procesar resultados web
                if isinstance(result, list):
                    for doc in result:
                        if isinstance(doc, dict) and 'url' in doc:
                            if doc['url'] not in docs:
                                docs[doc['url']] = doc
                                docs_str += f"\n- {doc.get('title', 'Sin título')}"
                                docs_str += f"\n  URL: {doc['url']}"
                                if 'score' in doc:
                                    docs_str += f"\n  Relevancia: {doc['score']:.2f}"
            else:
                # Para otros resultados (como analyze_sanctions_data)
                docs_str += str(result)
            
            msgs.append(ToolMessage(content=str(result), tool_call_id=tool_call["id"]))
            print(f"  Completado {tool_call['name']}")
            
            # Mostrar resultados parciales
            print("\nResultados parciales:")
            print(docs_str)
            
        except Exception as e:
            print(f"Error ejecutando {tool_call['name']}: {str(e)}")
            msgs.append(ToolMessage(content=f"Error: {str(e)}", tool_call_id=tool_call["id"]))
    
    return {"messages": msgs, "documents": docs}

def call_model(state: ResearchState):
    """Main analysis node"""
    prompt = f"""
    Eres un analista experto en riesgo y cumplimiento, especializado en debida diligencia corporativa y análisis de sanciones.
    
    TAREA: Analizar el perfil de riesgo de: {state['company_name']}

    CONTEXTO:
    - Has analizado una base de datos de sanciones y encontrado posibles coincidencias
    - Has realizado una búsqueda web en fuentes oficiales
    - Debes proporcionar un análisis detallado y recomendaciones

    Por favor, estructura tu análisis en español de la siguiente manera:

    1. ANÁLISIS DE COINCIDENCIAS EN LISTAS DE SANCIONES
    - Para cada coincidencia encontrada, analiza:
      * Relevancia y certeza de la coincidencia
      * Relación con la persona/entidad buscada
      * Implicaciones desde perspectiva de riesgo
      * Jurisdicciones involucradas

    2. ANÁLISIS DE INFORMACIÓN ADICIONAL
    - Analiza la información de fuentes oficiales encontrada
    - Identifica patrones o conexiones relevantes
    - Destaca elementos que aumentan o mitigan el riesgo

    3. EVALUACIÓN DE RIESGO
    - Nivel de riesgo: BAJO, MEDIO, ALTO o CRÍTICO
    - Factores clave que determinan el nivel de riesgo:
      * Gravedad de las coincidencias encontradas
      * Actualidad de la información
      * Jurisdicciones involucradas
      * Contexto general
    - Justificación detallada del nivel asignado

    4. RECOMENDACIONES
    - Acciones inmediatas requeridas
    - Medidas de debida diligencia adicional
    - Documentación específica a solicitar
    - Controles o monitoreo recomendados

    IMPORTANTE:
    - Sé específico y detallado en tu análisis
    - Fundamenta tus conclusiones en la evidencia encontrada
    - Proporciona recomendaciones accionables y concretas
    - Mantén un enfoque objetivo y profesional
    """
    
    print("\nIniciando análisis detallado...")
    messages = state['messages'] + [SystemMessage(content=prompt)]
    response = model.invoke(messages)
    print("Análisis completado")
    return {"messages": [response]}

def should_continue(state: ResearchState) -> Literal["tools", "output"]:
    """Determine workflow path"""
    messages = state['messages']
    last_message = messages[-1]
    
    # Si no hay tool_calls o ya se han procesado más de 3 iteraciones, terminar
    if not hasattr(last_message, 'tool_calls') or not last_message.tool_calls:
        return "output"
    
    # Contar cuántas veces se han ejecutado herramientas
    tool_executions = sum(1 for msg in messages if hasattr(msg, 'tool_calls') and msg.tool_calls)
    if tool_executions >= 3:
        print("\nLímite de iteraciones alcanzado, finalizando análisis...")
        return "output"
    
    return "tools"

def output_node(state: ResearchState):
    """Final output processing"""
    return state

# Define workflow
workflow = StateGraph(ResearchState)
workflow.add_node("research", call_model)
workflow.add_node("tools", tool_node)
workflow.add_node("output", output_node)

workflow.set_entry_point("research")
workflow.add_conditional_edges("research", should_continue)
workflow.add_edge("tools", "research")
workflow.add_edge("output", END)

graph = workflow.compile()

async def analyze_company(company_name: str):
    """
    Main function to analyze a company
    
    Args:
        company_name (str): Name of the company to analyze
        
    Returns:
        dict: Complete analysis results
    """
    print(f"\n{'='*50}")
    print(f"Iniciando análisis para: {company_name}")
    print(f"{'='*50}\n")
    
    initial_state = {
        "company_name": company_name,
        "sanctions_data": {},
        "documents": {},
        "messages": []
    }
    
    try:
        print("1. ANÁLISIS DE SANCIONES")
        print("------------------------")
        sanctions_result = await tools_by_name["analyze_sanctions_data"].ainvoke({"company_name": company_name})
        initial_state["sanctions_data"] = sanctions_result
        
        print("\n2. INVESTIGACIÓN WEB")
        print("--------------------")
        print(f"Realizando búsqueda web para: {company_name}")
        
        web_queries = [
            TavilyQuery(
                query=f"{company_name} sanctions OFAC SDN",
                topic="general",
                days=365,
                raw_content=True,
                include_domains=[
                    # OFAC y Treasury
                    "https://www.treasury.gov/",
                    "https://sanctionssearch.ofac.treas.gov/",
                    "https://home.treasury.gov/",
                    "https://ofac.treasury.gov/"
                ]
            ),
            TavilyQuery(
                query=f"{company_name} UIAF Colombia investigación",
                topic="news",
                days=180,
                raw_content=True,
                include_domains=[
                    # Colombia
                    "https://www.uiaf.gov.co/",
                    "https://www.supersociedades.gov.co/",
                    "https://www.fiscalia.gov.co/"
                ]
            )
        ]
        
        web_result = await tools_by_name["web_research"].ainvoke({"sub_queries": web_queries})
        initial_state["documents"] = {"web_results": web_result}
        
        print("\n3. ANÁLISIS FINAL")
        print("----------------")
        result = await graph.ainvoke(initial_state)
        
        if not result.get("messages"):
            print("ADVERTENCIA: No se generaron mensajes en el análisis")
            
        return {
            "company_name": company_name,
            "sanctions_data": sanctions_result,
            "documents": result.get("documents", {}),
            "messages": result.get("messages", []),
            "analysis_complete": True
        }
    except Exception as e:
        print(f"\nError durante el análisis: {str(e)}")
        import traceback
        print(f"Traceback completo:\n{traceback.format_exc()}")
        return {
            "error": str(e),
            "company_name": company_name,
            "analysis_complete": False
        }

if __name__ == "__main__":
    async def main():
        company = "Casa Grajales"
        try:
            result = await analyze_company(company)
            
            print("\n4. RESULTADOS FINALES")
            print("--------------------")
            
            if result.get("error"):
                print(f"\nEl análisis encontró errores: {result['error']}")
                return
            
            print("\nAnálisis completado exitosamente")
            
            # Mostrar resultados de sanciones
            sanctions_data = result.get("sanctions_data", {})
            if sanctions_data:
                print("\nA. RESULTADOS DE SANCIONES")
                print("=" * 50)
                print(f"Total de coincidencias: {sanctions_data.get('total_matches', 0)}")
                print(f"Entidades analizadas: {sanctions_data.get('total_entries_analyzed', 0)}")
                
                if "matches" in sanctions_data:
                    print("\nDetalles de coincidencias encontradas:")
                    for idx, match in enumerate(sanctions_data["matches"], 1):
                        print(f"\n{idx}. {match.get('entity_name', 'N/A')}")
                        print("-" * 50)
                        
                        # Nombres asociados (eliminando duplicados y ordenando)
                        all_names = sorted(set(match.get('all_names', [])))
                        if all_names:
                            print("\n   Nombres asociados:")
                            for name in all_names:
                                print(f"   • {name}")
                        
                        # Detalles de sanciones
                        details = match.get('sanctions_details', {})
                        
                        if details.get('programs'):
                            print("\n   Programas de sanciones:")
                            for program in sorted(set(details['programs'])):
                                print(f"   • {program}")
                        
                        if details.get('addresses'):
                            print("\n   Ubicaciones registradas:")
                            for addr in sorted(set(details['addresses'])):
                                print(f"   • {addr}")
                        
                        if details.get('id_numbers'):
                            print("\n   Documentos de identidad:")
                            for doc in sorted(set(details['id_numbers'])):
                                print(f"   • {doc}")
                        
                        # Información adicional
                        additional = details.get('additional_info', {})
                        if any(additional.values()):
                            print("\n   Información adicional:")
                            for key, value in additional.items():
                                if value:
                                    print(f"   • {key.title()}: {value}")
            
            # Mostrar resultados de búsqueda web
            web_results = result.get("documents", {}).get("web_results", [])
            if web_results:
                print("\nB. RESULTADOS DE BÚSQUEDA WEB")
                print("=" * 50)
                print(f"Total de fuentes encontradas: {len(web_results)}")
                
                # Agrupar por tipo de fuente
                ofac_results = []
                uiaf_results = []
                other_results = []
                
                for doc in web_results:
                    url = doc.get('url', '').lower()
                    if 'treasury.gov' in url or 'ofac' in url:
                        ofac_results.append(doc)
                    elif 'uiaf' in url or 'supersociedades' in url or 'fiscalia' in url:
                        uiaf_results.append(doc)
                    else:
                        other_results.append(doc)
                
                def print_results(title, results):
                    if results:
                        print(f"\n{title}")
                        print("-" * 50)
                        for idx, doc in enumerate(results, 1):
                            print(f"\n   {idx}. {doc.get('title', 'Sin título')}")
                            print(f"      URL: {doc.get('url', 'N/A')}")
                            if doc.get('score'):
                                print(f"      Relevancia: {doc['score']:.2f}")
                            if doc.get('content'):
                                content = doc['content'][:300]
                                print(f"      Resumen: {content}...")
                
                print_results("Fuentes OFAC/Treasury", ofac_results)
                print_results("Fuentes Colombia (UIAF/SuperSociedades)", uiaf_results)
                print_results("Otras fuentes relevantes", other_results)
            
            # Mostrar análisis de riesgo
            messages = result.get("messages", [])
            if messages:
                print("\nC. ANÁLISIS DE RIESGO")
                print("=" * 50)
                for msg in messages:
                    if hasattr(msg, "content") and not isinstance(msg, ToolMessage):
                        print(f"\n{msg.content}")
            
            print(f"\n{'='*50}")
            print("Análisis completado")
            print(f"{'='*50}")
            
        except Exception as e:
            print(f"\nError crítico durante la ejecución: {e}")
            import traceback
            print(f"Traceback completo:\n{traceback.format_exc()}")
    
    # Run the async main function
    asyncio.run(main()) 