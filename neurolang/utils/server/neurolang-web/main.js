import './style.css'
import $ from './jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import 'semantic-ui-css'
import { QueryController } from './query/query'
import { Router } from './router/router'
import { EnginesController } from './engines/engines'

dt(window, $)

const qc = new QueryController()
const ec = new EnginesController(qc)

const routes = [
  { path: '/', onRouteChange: (rp) => ec.setRouteEngine(rp) },
  { path: '/:engine', onRouteChange: (rp) => ec.setRouteEngine(rp) }
]

const router = new Router(routes)
