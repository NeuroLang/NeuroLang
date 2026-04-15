import { API_ROUTE } from '../constants'
import $ from '../jquery-bundler'
import './engines.css'
import showdown from 'showdown'

export class EnginesController {
  constructor (qc) {
    this.engine = null
    this.queryId = null
    this.qc = qc
    this.engines = null
    this.ready = false
    this.init()
  }

  init () {
    $.get(API_ROUTE.engines)
      .done((data) => {
        if ('status' in data && data.status === 'ok') {
          // Fetching the engines done with success
          this.engines = data.data
          this.initEnginesMenu()
          this.ready = true
          this.onRouteEngineChange()
        } else {
          this.onFail()
        }
      }).fail(() => this.onFail())
    const queryAccordion = $('.nl-query-info')
    queryAccordion.accordion()
  }

  onFail () {
    // An error occurred while fetching the engines.
    this.engines = undefined
    console.log('An error occurred while fetching the list of engines.')
    $('.nl-server-error-modal').modal('show')
  }

  /**
   * Called by the Router when a route change occurs.
   * @param {*} routeParams
   */
  setRouteEngine (routeParams) {
    this.engine = routeParams.engine
    this.queryId = routeParams.queryId
    if (this.ready) {
      this.onRouteEngineChange()
    }
  }

  /**
   * Set the new selected engine and query on the QueryController
   */
  onRouteEngineChange () {
    let selectedEngine = this.engines.find(e => e.engine === this.engine)
    if (!selectedEngine) {
      selectedEngine = this.engines[0]
    }
    let selectedQuery = selectedEngine.queries.find(q => q.id === this.queryId)
    if (!selectedQuery) {
      selectedQuery = selectedEngine.queries[0]
    }
    this.qc.setRouteEngine(selectedEngine.engine, selectedQuery.query)
    this.updateQueryHelp(selectedQuery)
  }

  /**
   * Initialize the dropdown menu with the list of engines
   * and queries.
   */
  initEnginesMenu () {
    const menu = $('.nl-dataset-dropdown')
    menu.empty()
    for (const engine of this.engines) {
      const item = $(`<a class="item" href="/${engine.engine}" data-link></a>`)
      if ('queries' in engine && engine.queries.length > 1) {
        $('<i class="dropdown icon"></i>').appendTo(item)
        $(`<span class="text">${engine.engine}</span>`).appendTo(item)
        const submenu = $('<div class="menu"></div>')
        engine.queries.forEach((q, i) => {
          $(`<a class="item" href="/${engine.engine}/${q.id}" data-link>${q.shortTitle}</a>`).appendTo(submenu)
        })
        submenu.appendTo(item)
      } else {
        item.text(engine.engine)
      }
      menu.append(item)
    }

    // show dropdown on hover
    $('.main.menu  .ui.dropdown').dropdown({
      on: 'hover'
    })
  }

  updateQueryHelp (selectedQuery) {
    const queryTitle = $('.nl-query-info .nl-query-title')
    queryTitle.text(selectedQuery.title)
    const queryDescription = $('.nl-query-info .nl-query-description')
    if ('description' in selectedQuery && selectedQuery.description) {
      const converter = new showdown.Converter()
      queryDescription.html(converter.makeHtml(selectedQuery.description))
    } else {
      queryDescription.html('')
    }
  }
}
