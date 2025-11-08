from src.service.Kg import KnowledgeGraph
from src.service.agents import IngestionAgent, RetrievalAgent
from src.service.manager import ManagerLLM
from src.service.weighting import SimpleWeightingModel, get_default_weighting_model


def bootstrap_demo_graph(ingestion: IngestionAgent) -> None:
    ingestion.store_concept('ApiFunction', 'API Function', parent_id=None)
    ingestion.store_concept('API_v1', 'API v1', parent_id='ApiFunction')
    ingestion.store_concept('API_v2', 'API v2', parent_id='ApiFunction')
    ingestion.update_concept('API_v1', 'API_v2', relation='REPLACED_BY')
    ingestion.store_concept(
        'route_session',
        'route /session',
        parent_id='API_v1',
        relation='HAS_ROUTE',
        properties={'documented_in': 'doc_api_v1.md'},
        link_direction='parent_to_child'
    )


def main() -> None:
    kg = KnowledgeGraph()
    manager = ManagerLLM()
    ingestion_agent = IngestionAgent(kg)
    weighting_model = get_default_weighting_model(kg)
    retrieval_agent = RetrievalAgent(kg, weighting_model)

    kg.clear()
    bootstrap_demo_graph(ingestion_agent)

    user_query = 'Historical check: did API v1 include the /session route?'
    instruction = manager.analyze(user_query)
    results = retrieval_agent.retrieve(instruction)
    for result in results:
        print('Entry:', result['entry_label'])
        for step in result['path']:
            print(
                f"  - {step['from_label']} -[{step['predicate']}]> {step['to_label']} (score={step['score']:.2f})"
            )
    kg.save()


if __name__ == '__main__':
    main()
