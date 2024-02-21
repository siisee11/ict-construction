import os
import pandas as pd
from llama_index.core.schema import TextNode, NodeRelationship, RelatedNodeInfo
from core.constants import INQUIRY_INDEX_PERSIST_DIR, INQUERY_DATA_PATH
from llama_index.core import VectorStoreIndex
from llama_index.core import StorageContext, load_index_from_storage


def load_inquiry_index():
    if not os.path.exists(INQUIRY_INDEX_PERSIST_DIR):
        df = pd.read_csv(INQUERY_DATA_PATH)
        nodes = []
        for index, row in df.iterrows():
            q_node = TextNode(
                id_=f"question_{index}",
                text=row["question"],
                metadata={
                    "filename": "정보통신공사업 질의회신 사례집",
                    "label": f"{index}번 질의",
                    "type": "question",
                },
            )

            a_node = TextNode(
                id_=f"answer_{index}",
                text=row["answer"],
                metadata={
                    "filename": "정보통신공사업 질의회신 사례집",
                    "label": f"{index}번 답변",
                    "type": "answer",
                },
            )

            q_node.relationships[NodeRelationship.NEXT] = RelatedNodeInfo(
                node_id=a_node.id_,
            )

            a_node.relationships[NodeRelationship.PREVIOUS] = RelatedNodeInfo(
                node_id=q_node.id_,
            )

            nodes.append(q_node)
            nodes.append(a_node)

        index = VectorStoreIndex(nodes)
        index.storage_context.persist(persist_dir=INQUIRY_INDEX_PERSIST_DIR)
    else:
        index = load_index_from_storage(
            StorageContext.from_defaults(persist_dir=INQUIRY_INDEX_PERSIST_DIR),
        )

    return index
