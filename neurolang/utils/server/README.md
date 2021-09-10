# NeuroLang Server

This directory contains an implementation of a web interface for Neurolang.

## Architecture

The application is composed of a backend server (based on [tornado](https://www.tornadoweb.org/en/stable/)), located in this directory, and a frontend application in HTML + JavaScript, located in the [neurolang-web](neurolang-web) directory.

## Requirements

To install the backend server, follow the setup instructions to install Neurolang.

To be able to build the frontend application, you need to install [Node.js](https://nodejs.org/en/) version >=12.0.0.

## Building

The frontend application can be built for production by running the command

```bash
$ npm install
$ npm run build
```

from the `neurolang-web` directory. This command will process the various javascript, html and css files and bundle them together in production-ready files. The `npm run build` command will output the built files in a `dist` directory.

The backend [server application](app.py) is configured to serve static files from this directory, so that if you build the frontend application with `npm run build -- --mode dev` and then start the backend server, you should be able to see the built frontend application by navigating to [http://localhost:8888/](http://localhost:8888/) (note the port, 8888, which is the one for the tornado server and not the one used by npm's development frontend server) (also note that we build the application with a specific mode `-- --mode dev`, since by default running `npm run build` will build the application for production, replacing the base api url with the url for the production server, i.e. `http://neurolang-u18.saclay.inria.fr, which would not work on a local machine).

For convenience, if neurolang was installed from source and npm is available in your path, the setup.py script builds the frontend by calling the `npm install & npm run build -- --mode dev` command. Once you start the tornado server (using `neuro-server` command or `python -m neurolang.utils.server.app`), the built application will then be available from [http://localhost:8888/](http://localhost:8888/).

## Development

Once Neurolang has been installed in a python environment, the server can be started by running
```bash
$ neuro-server
```
from a terminal with the python environment active.

Alternatively, the server can also be started with this command :
```bash
$ python -m neurolang.utils.server.app
```
executed from the root directory of the Neurolang project.

You can specify some command line options to the `neuro-server` executable. Notably, `--port` lets you specify the port on which the tornado server will listen, while `--data-dir` lets you specify the directory where downloaded datasets (Neurosynth, etc.) will be stored.

Run `neuro-server --help` for a full list of options.

To serve the frontend application, run the following commands to install the required libraries and start a development server :

```bash
$ cd neurolang-web
$ npm install
$ npm run dev
```

The frontend application will then be available from your browser at [http://localhost:3000/](http://localhost:3000/). The frontend will communicate with the backend server which is available on [http://localhost:8888/](http://localhost:8888/).

## Environment variables

Constants used by the frontend application are defined in the [constants.js](neurolang-web/constants.js) file. The value for some of these variables will depend on the environment in which the frontend application is served (dev, stage or prod), allowing for changing the behaviour of the app depending on the environment (for instance the API_URL can be `localhost` in development, and `http://my-production-site.fr` in production).

By default, if you run the frontend application with `npm run dev`, the mode will be `development`. If you build the application using `npm run build`, the mode will be production. To build the application for a specific mode, run `npm run build -- --mode stage` (here mode will be `stage`).

## Testing

Both parts of the application include unit tests. To run the tests for the frontend javascript code, run

```
$ npm test
```

from the `neurolang-web` directory.

Tests for the python tornado application can be run with pytest in the same way as other tests for Neurolang.
