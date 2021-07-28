# NeuroLang Server

This directory contains an implementation of a web interface for Neurolang.

## Architecture

The application is composed of a backend server (based on [tornado](https://www.tornadoweb.org/en/stable/)), located in this directory, and a frontend application in HTML + JavaScript, located in the [neurolang-web](neurolang-web) directory.

## Requirements

To install the backend server, follow the setup instructions to install Neurolang.

To be able to build the frontend application, you need to install [Node.js](https://nodejs.org/en/) version >=12.0.0.

## Usage

Once Neurolang has been installed in a python environment, the server can be started by running
```bash
$ neuro-server
```
from a terminal with the python environment active.

Alternatively, the server can also be started with this command :
```bash
$ python neurolang/utils/server/app.py
```
executed from the root directory of the Neurolang project.

To serve the frontend application, run the following commands to install the required libraries and start a development server :

```bash
$ cd neurolang-web
$ npm install
$ npm run dev
```

The frontend application will then be available from your browser at [http://localhost:3000/](http://localhost:3000/). The frontend will communicate with the backend server which is available on [http://localhost:8888/](http://localhost:8888/).