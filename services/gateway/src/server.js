// Start the gateway:  npm start   (from services/gateway)  ->  http://127.0.0.1:8200
import http from "node:http";
import mongoose from "mongoose";
import { attachLiveSocket, config, createApp } from "./app.js";
import { resumeWatchers } from "./watcher.js";

await mongoose.connect(config.mongoUri, { serverSelectionTimeoutMS: 5000 });
const server = http.createServer(createApp());
attachLiveSocket(server);
const resumed = await resumeWatchers();
server.listen(config.port, config.host, () => {
  console.log(`FraudNet-Synth gateway on http://${config.host}:${config.port}  (MongoDB: ${mongoose.connection.name}, `
              + `orchestrator: ${config.orchestratorUrl}, resumed watchers: ${resumed})`);
});
