import './style.css'
import $ from './jquery-bundler'
import 'datatables.net'
import dt from 'datatables.net-se'
import 'semantic-ui-css'
import { QueryController } from './query/query'

dt(window, $)

const qc = new QueryController()
