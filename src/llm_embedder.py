import time
import numpy as np
from tqdm import tqdm
from openai import OpenAI # 使用新的openai v1.x.x SDK

# Hugging Face Sentence Transformers (如果使用本地模型)
# try:
#     from sentence_transformers import SentenceTransformer
# except ImportError:
#     SentenceTransformer = None

class LLMEmbedder:
    def __init__(self, provider="openai", model_name="text-embedding-ada-002", api_key=None, batch_size=16):
        self.provider = provider
        self.model_name = model_name
        self.api_key = api_key
        self.batch_size = batch_size
        self.client = None

        if self.provider == "openai":
            if not self.api_key or self.api_key == "YOUR_API_KEY_HERE":
                raise ValueError("OpenAI API key 未配置或无效。请在 src/config.py 中设置。")
            self.client = OpenAI(api_key=self.api_key)
            print(f"已初始化 OpenAI embedder, 模型: {self.model_name}")
        # elif self.provider == "huggingface_local":
        #     if SentenceTransformer is None:
        #         raise ImportError("请安装 sentence-transformers: pip install sentence-transformers")
        #     try:
        #         self.client = SentenceTransformer(self.model_name)
        #         print(f"已初始化 Hugging Face SentenceTransformer, 模型: {self.model_name}")
        #     except Exception as e:
        #         raise RuntimeError(f"加载本地模型 {self.model_name} 失败: {e}")
        else:
            raise ValueError(f"不支持的LLM提供商: {self.provider}. 支持 'openai' 或 'huggingface_local'.")

    def get_embeddings(self, text_sequences):
        """
        获取一批文本序列的嵌入。

        Args:
            text_sequences (list of str): 文本序列列表。

        Returns:
            np.array: 形状为 (num_sequences, embedding_dim) 的嵌入向量。
        """
        all_embeddings = []
        print(f"正在使用 {self.provider} ({self.model_name}) 获取文本嵌入...")
        
        for i in tqdm(range(0, len(text_sequences), self.batch_size), desc="获取嵌入"):
            batch_texts = text_sequences[i:i + self.batch_size]
            
            if self.provider == "openai":
                try:
                    response = self.client.embeddings.create(
                        input=batch_texts,
                        model=self.model_name
                    )
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)
                except Exception as e:
                    print(f"OpenAI API调用失败: {e}")
                    print("请检查您的API密钥、网络连接和模型配额。")
                    # 对于失败的批次，可以填充None或零向量，或者直接抛出异常
                    # 这里为了简单，如果API调用失败，我们用零向量填充，并打印错误
                    # 更好的做法是加入重试机制
                    error_embedding_dim = 1536 # ada-002的维度, 需要根据模型调整
                    all_embeddings.extend([np.zeros(error_embedding_dim)] * len(batch_texts))
                    time.sleep(1) # 避免频繁失败请求

            # elif self.provider == "huggingface_local":
            #     try:
            #         batch_embeddings = self.client.encode(batch_texts, show_progress_bar=False)
            #         all_embeddings.extend(batch_embeddings.tolist()) # SentenceTransformer 返回numpy数组
            #     except Exception as e:
            #         print(f"本地模型 ({self.model_name}) 推理失败: {e}")
            #         # 获取本地模型的输出维度
            #         try:
            #             error_embedding_dim = self.client.get_sentence_embedding_dimension()
            #         except:
            #             error_embedding_dim = 768 # 一个通用的默认值
            #         all_embeddings.extend([np.zeros(error_embedding_dim)] * len(batch_texts))
            
            # 礼貌性等待，避免API速率限制 (对某些API可能需要)
            # time.sleep(0.1) 

        if not all_embeddings:
             print("错误：未能获取任何嵌入向量。")
             return np.array([])
             
        return np.array(all_embeddings)

if __name__ == '__main__':
    from config import OPENAI_API_KEY, LLM_PARAMS

    if OPENAI_API_KEY == "YOUR_API_KEY_HERE":
        print("警告: OpenAI API 密钥未配置。跳过 LLMEmbedder 测试。")
    else:
        test_texts = [
            "0 1 0 , 0 2 0 , 0 3 0",
            "1 0 0 , 0 5 0 , 0 0 1",
            "7 5 3 , 1 2 0 , 0 9 5"
        ]
        
        # 测试 OpenAI
        try:
            openai_embedder = LLMEmbedder(provider="openai", 
                                          model_name=LLM_PARAMS['model_name'], 
                                          api_key=OPENAI_API_KEY,
                                          batch_size=2)
            openai_embeddings = openai_embedder.get_embeddings(test_texts)
            if openai_embeddings.size > 0:
                 print("\nOpenAI 嵌入 (前2个):")
                 print(openai_embeddings[:2])
                 print("嵌入维度:", openai_embeddings.shape)
            else:
                print("未能获取OpenAI嵌入。")

        except ValueError as e:
            print(f"OpenAI Embedder 初始化失败: {e}")
        except Exception as e:
            print(f"OpenAI Embedder 测试时发生未知错误: {e}")

        # # 测试 Hugging Face 本地模型 (需要安装 sentence-transformers 并下载模型)
        # print("\n--- Hugging Face 本地模型测试 ---")
        # try:
        #     hf_embedder = LLMEmbedder(provider="huggingface_local",
        #                               model_name="sentence-transformers/all-MiniLM-L6-v2", # 这是一个常用的小型模型
        #                               batch_size=2)
        #     hf_embeddings = hf_embedder.get_embeddings(test_texts)
        #     if hf_embeddings.size > 0:
        #         print("\nHugging Face 嵌入 (前2个):")
        #         print(hf_embeddings[:2])
        #         print("嵌入维度:", hf_embeddings.shape)
        #     else:
        #         print("未能获取Hugging Face嵌入。")
        # except ImportError as e:
        #     print(f"Hugging Face 测试跳过: {e}")
        # except RuntimeError as e:
        #      print(f"Hugging Face Embedder 初始化或推理失败: {e}")
        # except Exception as e:
        #     print(f"Hugging Face Embedder 测试时发生未知错误: {e}")