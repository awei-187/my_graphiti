import asyncio
import json
import logging
import os
from collections import defaultdict
from time import time
from logging import INFO

from dotenv import load_dotenv

from graphiti_core import Graphiti
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_generic_client import OpenAIGenericClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.nodes import EntityNode
from graphiti_core.edges import EntityEdge
from graphiti_core.search.search_config_recipes import EDGE_HYBRID_SEARCH_CROSS_ENCODER, NODE_HYBRID_SEARCH_RRF
import asyncio

#################################################
# CONFIGURATION
#################################################
# Set up logging and environment variables for
# connecting to Neo4j database
#################################################

# Configure logging
logging.basicConfig(
    level=INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S',
)
logger = logging.getLogger(__name__)

load_dotenv()

# Neo4j connection parameters
# Make sure Neo4j Desktop is running with a local DBMS started
neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
neo4j_password = os.environ.get('NEO4J_PASSWORD')

if not neo4j_uri or not neo4j_user or not neo4j_password:
    raise ValueError('NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set')

# Configure Ollama LLM client
llm_config = LLMConfig(
    api_key = os.environ.get('OPENAI_API_KEY'),  # Ollama doesn't require a real API key, but some placeholder is neededd
    base_url = os.environ.get('OPENAI_BASE_URL'),  # Ollama's OpenAI-compatible endpoint
)

llm_client = OpenAIGenericClient(config=llm_config)

# Initialize Graphiti
graphiti = Graphiti(
    neo4j_uri,
    neo4j_user,
    neo4j_password,
    llm_client=llm_client,
    embedder=OpenAIEmbedder(
        config=OpenAIEmbedderConfig(
            api_key=os.environ.get('OPENAI_API_KEY'),  # Placeholder API key
            base_url=os.environ.get('OPENAI_BASE_URL'),
        )
    ),
    cross_encoder=OpenAIRerankerClient(client=llm_client, config=llm_config),
)

#################################################
# HELPER FUNCTIONS (与Zep脚本完全相同)
#################################################
TEMPLATE = """
FACTS and ENTITIES represent relevant context to the current conversation.

# These are the most relevant facts for the conversation along with the datetime of the event that the fact refers to.
If a fact mentions something happening a week ago, then the datetime will be the date time of last week and not the datetime
of when the fact was stated.
Timestamps in memories represent the actual time the event occurred, not the time the event was mentioned in a message.
    
<FACTS>
{facts}
</FACTS>

# These are the most relevant entities
# ENTITY_NAME: entity summary
<ENTITIES>
{entities}
</ENTITIES>
"""

def compose_search_context(edges: list[EntityEdge], nodes: list[EntityNode]) -> str:
    facts = [f'  - {edge.fact} (event_time: {edge.valid_at})' for edge in edges]
    entities = [f'  - {node.name}: {node.summary}' for node in nodes]
    return TEMPLATE.format(facts='\n'.join(facts), entities='\n'.join(entities))


async def main():
    try:
        # Load LOCOMO JSON data
        with open('locomo10.json', 'r', encoding='utf-8') as f:
            locomo_data = json.load(f)

        graphiti_search_results = defaultdict(list)
        
        # 遍历所有用户/对话
        for conversation in locomo_data:
            # 在 Graphiti 中，我们使用 owner_id 来实现用户隔离
            owner_id = conversation.get('sample_id')
            qa_set = conversation.get('qa', [])
            
            logger.info(f"--- Processing searches for Owner ID: {owner_id} ---")

            for qa in qa_set:
                query = qa.get('question')
                if qa.get('category') == 5 or not query:
                    continue
                
                start = time()

                # --- 核心修改：使用 Graphiti 的方法进行并发搜索 ---
                # 1. 准备节点搜索的配置
                edge_search_config = EDGE_HYBRID_SEARCH_CROSS_ENCODER.model_copy(deep=True)
                edge_search_config.limit = 20
                node_search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
                node_search_config.limit = 20

                # 2. 使用 asyncio.gather 并发执行边搜索和节点搜索
                search_results = await asyncio.gather(
                    # 边 (Facts) 搜索
                    graphiti.search(
                        query=query, 
                        config=edge_search_config,
                        group_ids=[owner_id], # 注意：这里是 group_ids 列表
                    ),
                    # 节点 (Entities) 搜索
                    graphiti._search(
                        query=query,
                        config=node_search_config,
                        group_ids=[owner_id] # 同样传入 group_ids
                    )
                )

                # 3. 解析并发搜索的结果
                edges = search_results[0]  # graphiti.search 直接返回 list[EntityEdge]
                nodes = search_results[1].nodes # graphiti._search 返回一个带 .nodes 属性的对象
                # ----------------------------------------------------

                context = compose_search_context(edges, nodes)
                duration_ms = (time() - start) * 1000

                graphiti_search_results[owner_id].append({'context': context, 'duration_ms': duration_ms, 'question': query})

            logger.info(f"Finished all searches for Owner ID: {owner_id}")

        # 确保输出目录存在
        os.makedirs("data", exist_ok=True)

        # 保存结果到文件
        output_filepath = "data/graphiti_locomo_search_results.json"
        with open(output_filepath, "w", encoding='utf-8') as f:
            json.dump(dict(graphiti_search_results), f, indent=2)
            logger.info(f'Successfully saved Graphiti search results to {output_filepath}')

    finally:
        await graphiti.close()
        logger.info('\nNeo4j connection closed.')


if __name__ == "__main__":
    asyncio.run(main())