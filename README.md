# Multimodal AI Pricing & Recommendation Engine 🏷️✨

A scalable machine learning pipeline and API that generates multimodal embeddings (text + images) for fashion items, indexes them into a vector database, and calculates automated pricing suggestions and similarity recommendations.

---

### 🛠️ Tech Stack & Architecture

- **Language & Frameworks:** Python, FastAPI
- **Vector Search & Embeddings:** Qdrant Vector Database, OpenAI CLIP (`clip-vit-base-patch32`), PyTorch
- **Big Data Processing:** PySpark
- **Containerization & Deployment:** Docker, Docker Compose

---

### ✨ Key Features

- **Multimodal Vector Indexing:** Extracts deep visual and text features using CLIP transformers to generate 512-dimensional embeddings for fashion listings.
- **High-Performance Similarity Search:** Stores and queries high-dimensional vectors in **Qdrant** for instant semantic retrieval and product similarity matching.
- **Distributed Ingestion:** Leverages **PySpark** User-Defined Functions (UDFs) to process product datasets at scale.
- **Automated AI Pricing Suggestions:** Computes price estimates based on vector similarity metrics with historical market listings.
- **Containerized Environment:** Fully orchestrated with Docker Compose for seamless local development and deployment.
