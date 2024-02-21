import os
import pandas as pd
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from core.constants import (
    ACT_INDEX_PERSIST_DIR,
    ACT_DATA_PATH,
    ACT_ENFORCEMENT_DATA_PATH,
    ACT_ENFORCEMENT_INDEX_PERSIST_DIR,
)
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage


def load_act_index():
    if not os.path.exists(ACT_INDEX_PERSIST_DIR):
        df = pd.read_csv(ACT_DATA_PATH, sep="\t")
        curr_article_id = None
        curr_article_node = None
        nodes = []

        for index, row in df.iterrows():
            chapter_id = row["chapter_id"]
            article_id = row["article_id"]

            if curr_article_id != article_id:
                curr_article_id = article_id
                curr_article_node = TextNode(
                    id_=f"{chapter_id}_{article_id}",
                    text=row["article_content"],
                    metadata={
                        "filename": "정보통신공사업법",
                        "label": f"{chapter_id}장 {article_id}조",
                        "chapter_title": row["chapter_title"],
                        "article_title": row["article_title"],
                        "chapter_id": chapter_id,
                        "article_id": article_id,
                    },
                    excluded_llm_metadata_keys=["chapter_id", "article_id"],
                )
                nodes.append(curr_article_node)

            node = TextNode(
                id_=f"{chapter_id}_{article_id}_{index}",
                text=row["content"],
                metadata={
                    "filename": "정보통신공사업법",
                    "label": f"{chapter_id}장 {article_id}조",
                    "chapter_title": row["chapter_title"],
                    "article_title": row["article_title"],
                    "chapter_id": chapter_id,
                    "article_id": article_id,
                },
                excluded_llm_metadata_keys=["chapter_id", "article_id"],
            )

            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=curr_article_node.id_,
            )

            nodes.append(node)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=ACT_INDEX_PERSIST_DIR)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=ACT_INDEX_PERSIST_DIR),
        )

    return index


def load_act_enforcement_index():
    if not os.path.exists(ACT_ENFORCEMENT_INDEX_PERSIST_DIR):
        df = pd.read_csv(ACT_ENFORCEMENT_DATA_PATH, sep="\t")
        curr_article_id = None
        curr_article_node = None
        nodes = []

        for index, row in df.iterrows():
            chapter_id = row["chapter_id"]
            article_id = row["article_id"]

            if curr_article_id != article_id:
                curr_article_id = article_id
                curr_article_node = TextNode(
                    id_=f"{chapter_id}_{article_id}",
                    text=row["article_content"],
                    metadata={
                        "filename": "정보통신공사업법 시행령",
                        "label": f"{chapter_id}장 {article_id}조",
                        "chapter_title": row["chapter_title"],
                        "article_title": row["article_title"],
                        "chapter_id": chapter_id,
                        "article_id": article_id,
                    },
                    excluded_llm_metadata_keys=["chapter_id", "article_id"],
                )
                print(curr_article_node)
                nodes.append(curr_article_node)

            node = TextNode(
                id_=f"{chapter_id}_{article_id}_{index}",
                text=row["content"],
                metadata={
                    "filename": "정보통신공사업법 시행령",
                    "label": f"{chapter_id}장 {article_id}조",
                    "chapter_title": row["chapter_title"],
                    "article_title": row["article_title"],
                    "chapter_id": chapter_id,
                    "article_id": article_id,
                },
                excluded_llm_metadata_keys=["chapter_id", "article_id"],
            )

            node.relationships[NodeRelationship.PARENT] = RelatedNodeInfo(
                node_id=curr_article_node.id_,
            )

            nodes.append(node)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=ACT_ENFORCEMENT_INDEX_PERSIST_DIR)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=ACT_ENFORCEMENT_INDEX_PERSIST_DIR),
        )

    return index
