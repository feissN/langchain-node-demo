import { OpenAI } from 'langchain/llms/openai';
import { RetrievalQAChain } from 'langchain/chains';
import { HNSWLib } from 'langchain/vectorstores/hnswlib';
import { OpenAIEmbeddings } from 'langchain/embeddings/openai';
import { RecursiveCharacterTextSplitter } from 'langchain/text_splitter';
import { DirectoryLoader } from 'langchain/document_loaders/fs/directory';
import * as fs from 'fs';
import * as dotenv from 'dotenv';
import { CustomPDFLoader } from './utils/customPDFLoader';

dotenv.config();

const filePath = "docs";

const question = "Where can I find the JS documentation?";
const VECTOR_STORE_PATH = `my_new_store.index`;

export const runWithEmbeddings = async () => {
    try {
        const model = new OpenAI({});

        let vectorStore: HNSWLib;
        if (fs.existsSync(VECTOR_STORE_PATH)) {
            console.log("Vector exists")
            vectorStore = await HNSWLib.load(VECTOR_STORE_PATH, new OpenAIEmbeddings());
        } else {
            console.log("Creating new Vector")

            const directoryLoader = new DirectoryLoader(filePath, {
                '.pdf': (path) => new CustomPDFLoader(path),
            });
            const rawDocs = await directoryLoader.load();
            const textSplitter = new RecursiveCharacterTextSplitter({
                chunkSize: 1000,
                chunkOverlap: 200,
            });

            const docs = await textSplitter.splitDocuments(rawDocs);
            vectorStore = await HNSWLib.fromDocuments(docs, new OpenAIEmbeddings());
            await vectorStore.save(VECTOR_STORE_PATH);
        }

        const chain = RetrievalQAChain.fromLLM(model, vectorStore.asRetriever());

        const res = await chain.call({
            query: question,
        });

        console.log({ question, response: res.text });
    } catch (error) {
        console.error("Error", error)
    }
};

runWithEmbeddings();
