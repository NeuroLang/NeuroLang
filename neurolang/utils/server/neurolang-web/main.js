import './style.css'
import $ from './jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import 'fomantic-ui-css/semantic'
import { QueryController } from './query/query'
import { Router } from './router/router'
import { EnginesController } from './engines/engines'

dt(window, $)

const qc = new QueryController()
// const ec = new EnginesController(qc)

window.qc = qc
const ec = new EnginesController(window.qc)

const routes = [
  { path: '/:engine/:queryId', onRouteChange: (rp) => ec.setRouteEngine(rp) },
  { path: '/:engine', onRouteChange: (rp) => ec.setRouteEngine(rp) },
  { path: '/', onRouteChange: (rp) => ec.setRouteEngine(rp) }
]

const router = new Router(routes)
