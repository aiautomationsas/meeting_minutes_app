from typing import Literal, Dict, List
from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from typing import Dict, List, TypedDict, Annotated
from dotenv import load_dotenv

load_dotenv()

model = ChatOpenAI(model="gpt-4o-mini", temperature=0)

@tool
def analyze_indicators() -> Dict[str, Dict]:
    """
    Indicadores disponibles para el programa Fábricas de Productividad.
    Returns:
        Diccionario con los tipos de indicadores y sus detalles para las líneas de:
        - Productividad Operacional
        - Transformación Digital
    """
    indicators = {
        "operacional_productivity": {
            "indicadores_fijos": {
                "TNVA": {
                    "nombre": "Reducción de tiempo de no valor agregado",
                    "unidad": "Minutos, horas, días, segundos",
                    "formula": "Sumatoria de Tiempos de no valor agregado para cada subproceso",
                    "naturaleza": "Disminuir",
                    "entregable": "Herramienta gráfica del proceso o subprocesos a mejorar"
                },
                "ahorros_desperdicios": {
                    "TCP": {
                        "nombre": "Tiempo de ciclo productivo",
                        "formula": "TCP * Valor por minuto",
                        "naturaleza": "Disminuir"
                    },
                    "CMP": {
                        "nombre": "Consumo de materia prima",
                        "formula": "Materia prima * Costo Unitario",
                        "naturaleza": "Disminuir"
                    },
                    "DE": {
                        "nombre": "Defectos",
                        "formula": "((Producción defectuosa)/(Producción total))*Costo de producción total",
                        "naturaleza": "Disminuir"
                    },
                    "OUM": {
                        "nombre": "Optimización uso de maquinaria",
                        "formula": "(1-(D*R*C)*100)*VMP",
                        "naturaleza": "Disminuir"
                    },
                    "CU": {
                        "nombre": "Costo Unitario",
                        "formula": "(costo total producción)/(número de productos)",
                        "naturaleza": "Disminuir"
                    },
                    "CCI": {
                        "nombre": "Costo de Calidad Interna",
                        "formula": "Σ valor desperdicios (reprocesos, retrabajos, sobrantes)",
                        "naturaleza": "Disminuir"
                    },
                    "CCE": {
                        "nombre": "Costo de Calidad Externa",
                        "formula": "Σ valor (devoluciones, garantías, servicios posventa)",
                        "naturaleza": "Disminuir"
                    }
                }
            },
            "indicadores_variables": {
                "TCP": {
                    "nombre": "Tiempos de ciclo productivo",
                    "requisito": True,
                    "entregable": "Herramienta gráfica del proceso"
                },
                "RE": {
                    "nombre": "Recuperación de espacios",
                    "formula": "(Metros recuperados / Total de metros cuadrados)*100",
                    "requisito": True,
                    "entregable": "Fotos y planos antes/después"
                },
                "TA": {
                    "nombre": "Tiempo de alistamiento promedio de equipos",
                    "formula": "∑ Tiempos de alistamiento equipos / Total de equipos intervenidos"
                },
                "NRI": {
                    "nombre": "Nivel de rotación de inventario",
                    "formula": "Costo de Ventas / (MP + PP + PT)"
                },
                "Pull_System": {
                    "nombre": "Nivel de Pull System",
                    "formula": "(Número de ítems de MP en sitio de proceso ÷ Número total de ítems de MP)*100"
                }
            },
            "indicadores_sostenibilidad": {
                "NCR": {
                    "nombre": "Nivel de generación de Residuos",
                    "formula": "Unidad de generación de residuos / Unidad producida o servicio prestado",
                    "naturaleza": "Disminuir"
                },
                "NCE": {
                    "nombre": "Nivel de consumo de Energía",
                    "formula": "Nivel de energía consumida / # Unidades producida",
                    "unidad": "kWh/mes",
                    "naturaleza": "Disminuir"
                },
                "IA_Combustible": {
                    "nombre": "Impacto Ambiental – Consumo de Combustible",
                    "formula": "Galón combus. usado x mes / Km recorridos x mes",
                    "unidad": "Galones/km",
                    "naturaleza": "Disminuir"
                }
            }
        },
        "transformacion_digital": {
            "indicadores_fijos": {
                "AT": {
                    "nombre": "Ahorros en tiempos por aplicación de herramientas digitales",
                    "formula": "AT = TEn",
                    "unidad": "Minutos",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Identificar tiempo efectivo del proceso antes de la tecnología (TEn)",
                        "Comparar tiempo antes vs. después",
                        "Calcular reducción de tiempo",
                        "Documentar en VSM o diagrama de flujo"
                    ]
                },
                "RTH": {
                    "nombre": "Reducción de costos de Talento Humano en procesos por Automatización",
                    "formula": "RTH = CHn * Tn",
                    "unidad": "Pesos Colombianos",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Determinar costo por hora de talento humano (CHn)",
                        "Calcular tiempo destinado a la tarea (Tn)",
                        "Multiplicar costo por tiempo",
                        "Comparar costos antes y después de automatización"
                    ]
                },
                "TMR": {
                    "nombre": "Tiempo de recuperación de procesos críticos por digitalización",
                    "formula": "TMR = MAX {TRP1, TRP2, TRP3...TRPn}",
                    "unidad": "Días",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Identificar procesos críticos",
                        "Evaluar tiempo de recuperación de cada proceso (TRP)",
                        "Seleccionar el tiempo máximo de recuperación",
                        "Documentar en matriz de riesgos"
                    ]
                }
            },
            "indicadores_sostenibilidad": {
                "IAD": {
                    "nombre": "Impacto Ambiental de Digitalización de procesos",
                    "formula": "IAD = Número de hojas en un mes",
                    "unidad": "Número de hojas",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Contabilizar hojas usadas mensualmente",
                        "Comparar consumo antes y después de digitalización",
                        "Preparar inventario de recursos",
                        "Documentar reducción de consumo"
                    ]
                },
                "ITD_residuos": {
                    "nombre": "Impacto Ambiental - Residuos",
                    "formula": "ITD = Unidad de Generación de residuos / Unidad producida",
                    "unidad": ["TON", "Kg", "Metros"],
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Medir total de residuos generados",
                        "Dividir por unidades producidas",
                        "Comparar ratio antes y después de implementación tecnológica",
                        "Analizar optimización de recursos"
                    ]
                },
                "ITD_agua": {
                    "nombre": "Impacto Ambiental - Consumo de Agua",
                    "formula": "ITD Agua = (Volumen agua consumida) / (Unidades producidas)",
                    "unidad": "Litros/Unidad",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Medir volumen total de agua consumida",
                        "Dividir por unidades producidas",
                        "Implementar sensores IoT para monitoreo",
                        "Comparar eficiencia de consumo"
                    ]
                },
                "ITD_energia": {
                    "nombre": "Impacto Ambiental - Consumo de Energía",
                    "formula": "ITD energía = (Volumen energía consumida) / (Unidades producidas)",
                    "unidad": "kWh/Unidad",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Registrar consumo total de energía",
                        "Dividir por unidades producidas",
                        "Usar sistemas inteligentes de monitoreo",
                        "Analizar optimización energética"
                    ]
                }
            },
            "indicadores_variables": {
                "AUT": {
                    "nombre": "Procesos optimizados por automatización",
                    "formula": "AUT = (PA / P) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Contar procesos automatizados (PA)",
                        "Calcular total de procesos (P)",
                        "Dividir PA/P y multiplicar por 100",
                        "Documentar procesos manuales eliminados"
                    ]
                },
                "CACN": {
                    "nombre": "Reducción Costos Adquisición de Clientes",
                    "formula": "CACN = CAC / CN",
                    "unidad": "Pesos por cliente",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Calcular costo total de adquisición (CAC)",
                        "Dividir por número de clientes nuevos (CN)",
                        "Comparar con costos históricos",
                        "Identificar mejoras en eficiencia"
                    ]
                },
                "TV": {
                    "nombre": "Tasa de Ventas",
                    "formula": "TV = (CV / CP) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Contabilizar clientes vendidos (CV)",
                        "Registrar clientes potenciales (CP)",
                        "Dividir CV/CP y multiplicar por 100",
                        "Analizar eficacia de canales"
                    ]
                },
                "TR": {
                    "nombre": "Tasa de Rebote",
                    "formula": "TR = (AP / VP) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Disminuir",
                    "metodologia": [
                        "Contar accesos a página única (AP)",
                        "Medir visitantes totales de página (VP)",
                        "Calcular porcentaje de abandonos",
                        "Mejorar diseño y contenido web"
                    ]
                },
                "TC": {
                    "nombre": "Tableros de Control",
                    "formula": "TC = TC",
                    "unidad": "Unidad",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Identificar tableros de control creados",
                        "Documentar datos mostrados",
                        "Registrar áreas beneficiadas",
                        "Evaluar impacto en toma de decisiones"
                    ]
                },
                "CI": {
                    "nombre": "Colaboraciones Internas",
                    "formula": "CI = (AII / TA) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Contar áreas con información integrada (AII)",
                        "Calcular total de áreas (TA)",
                        "Dividir AII/TA y multiplicar por 100",
                        "Documentar sistemas de información"
                    ]
                },
                "FTD": {
                    "nombre": "Formación Personal Digital",
                    "formula": "FTD = (HF / HL) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Registrar horas de formación (HF)",
                        "Calcular horas laborales totales (HL)",
                        "Dividir HF/HL y multiplicar por 100",
                        "Analizar desarrollo de competencias digitales"
                    ]
                },
                "PC": {
                    "nombre": "Participación de Colaboradores",
                    "formula": "PC = (CSI / TC) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Contar colaboradores usando Sistemas de Información (CSI)",
                        "Determinar total de colaboradores (TC)",
                        "Dividir CSI/TC y multiplicar por 100",
                        "Documentar sistemas utilizados"
                    ]
                },
                "CD": {
                    "nombre": "Clientes en Canales Digitales",
                    "formula": "CD = (CCD / CA) * 100",
                    "unidad": "Porcentaje",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Contar clientes en canales digitales (CCD)",
                        "Calcular clientes activos totales (CA)",
                        "Dividir CCD/CA y multiplicar por 100",
                        "Identificar canales digitales disponibles"
                    ]
                },
                "IT": {
                    "nombre": "Inversión en Tecnología",
                    "formula": "IT = Inversión en tecnología año Ti",
                    "unidad": "Pesos",
                    "naturaleza": "Aumentar",
                    "metodologia": [
                        "Registrar inversión en sistemas de información",
                        "Comparar con inversiones de períodos anteriores",
                        "Analizar crecimiento de inversión tecnológica",
                        "Documentar impacto de inversiones"
                    ]
                }
            }
        }
    }
    
    return indicators


class BusinessProposal(TypedDict):
    executive_summary: Annotated[str, "Brief summary of the problem and proposed solution"]
    company_description: Annotated[str, "Brief summary of the company"]
    problem_statement: Annotated[str, "Detailed explanation of cost reduction opportunities"]
    proposed_solution: Annotated[dict, {
        "strategies": "List of specific strategies and interventions including the indicators to be implemented",
        "timeline": "Implementation timetable",
        "resources": "Required resources for implementation",
        "kpis": {
            "fixed_indicators": "2 mandatory fixed indicators from the fixed indicators list",
            "variable_indicators": "4 variable indicators from the variable indicators list",
            "sustainability_indicator": "1 sustainability indicator from the sustainability indicators list",
            "measurement_details": {
                "indicator_name": "Name of the indicator",
                "current_value": "Current baseline value",
                "target_value": "Expected target value",
                "measurement_frequency": "How often it will be measured",
                "measurement_method": "How the indicator will be measured"
            }
        }
    }]
    financial_implications: Annotated[dict, {
        "implementation_costs": "Breakdown of costs",
        "roi_analysis": "Return on investment projections",
        "payback_period": "Expected time to recover investment"
    }]
    team_and_methodology: Annotated[dict, {
        "team_members": "List of team members and their roles",
        "methodology": "Project methodology and approach"
    }]
    conclusion: Annotated[str, "Summary of key points and benefits"]
    
@tool
def select_business_line() -> Literal["operacional_productivity", "transformacion_digital"]:
    """
    Pregunta al usuario para cuál línea desea generar la propuesta de negocio:
    - Productividad Operacional
    - Transformación Digital
    
    Returns:
        La línea de negocio seleccionada como un string literal
    """
    # Esta función será implementada por el agente para interactuar con el usuario
    pass

@tool
def generate_business_proposal(business_line: Literal["operacional_productivity", "transformacion_digital"]) -> BusinessProposal:
    """
    Genera una propuesta de negocio estructurada basada en la línea de negocio seleccionada.
    
    Args:
        business_line: Línea de negocio seleccionada ("operacional_productivity" o "transformacion_digital")
    
    La propuesta debe incluir exactamente:
    - 2 indicadores fijos de la lista de indicadores fijos
    - 4 indicadores variables de la lista de indicadores variables
    - 1 indicador de sostenibilidad de la lista de indicadores de sostenibilidad
    
    Los indicadores seleccionados deben corresponder a la línea de negocio especificada.
    
    Returns:
        Un objeto BusinessProposal con todas las secciones requeridas
    """
    
    proposal: BusinessProposal = {
        "executive_summary": "",
        "company_description": "",
        "problem_statement": "",
        "proposed_solution": {
            "strategies": "",
            "timeline": "",
            "resources": "",
            "kpis": {
                "fixed_indicators": [],  # Debe contener exactamente 2 indicadores
                "variable_indicators": [],  # Debe contener exactamente 4 indicadores
                "sustainability_indicator": "",  # Debe contener exactamente 1 indicador
                "measurement_details": {
                    "indicator_name": "",
                    "current_value": "",
                    "target_value": "",
                    "measurement_frequency": "",
                    "measurement_method": ""
                }
            }
        },
        "financial_implications": {
            "implementation_costs": "",
            "roi_analysis": "",
            "payback_period": ""
        },
        "team_and_methodology": {
            "team_members": "",
            "methodology": ""
        },
        "conclusion": ""
    }
    
    return proposal


tools = [analyze_indicators, select_business_line, generate_business_proposal]

# Creamos el agente
from langgraph.prebuilt import create_react_agent

graph = create_react_agent(
    model,
    tools,
    )