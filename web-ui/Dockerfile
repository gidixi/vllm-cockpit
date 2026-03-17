FROM node:18-slim

WORKDIR /app

# Copia package.json e installa dipendenze
COPY package.json .
RUN npm install

# Copia i file pubblici
COPY public ./public
COPY src ./src

# Assicura che tutte le dipendenze siano installate (inclusi react-markdown e remark-gfm)
RUN npm install

# Esponi la porta di React
EXPOSE 3000

# Avvia React in dev mode
CMD ["npm", "start"]
