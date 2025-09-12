FROM node:18-slim

WORKDIR /app
COPY package.json .
RUN npm install --only=prod
COPY app.js .

EXPOSE 3000
CMD ["npm", "start"]

