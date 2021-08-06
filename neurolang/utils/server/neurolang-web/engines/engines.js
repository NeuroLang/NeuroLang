import { API_ROUTE } from '../constants'
import $ from '../jquery-bundler'
import './engines.css'

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
    // $.get(API_ROUTE.engines)
    //   .done((data) => {
    //     this.engines = data
    //   })
    this.engines = [
      {
        engine: 'neurosynth',
        queries: [
          {
            title: 'Single term', id: 'neuro1', query: `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
ActivationGivenTerm(i, j, k, PROB) :- Activation(i, j, k) // TermAssociation("emotion")
ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)`
          },
          {
            title: 'Multiple term', id: 'neuro2', query: `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
TermsToSelect("emotion")
TermsToSelect("fear")
Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
ActivationGivenTerm(i, j, k, t, PROB) :- Activation(i, j, k) // (TermAssociation(t) & TermsToSelect(t))
ActivationGivenTermImage(t, agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, t, p)`
          }
        ]
      },
      {
        engine: 'destrieux',
        queries: [
          {
            title: 'Regions union', id: 'destrieux1', query: 'union(region_union(r)) :- destrieux(..., r)'
          }
        ]
      }
    ]
    this.initEnginesMenu()
    this.ready = true
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
          $(`<a class="item" href="/${engine.engine}/${q.id}" data-link>${q.title}</a>`).appendTo(submenu)
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
}
