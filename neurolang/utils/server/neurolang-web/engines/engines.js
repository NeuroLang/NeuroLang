import { API_ROUTE } from '../constants'
import $ from '../jquery-bundler'

export class EnginesController {
  constructor (qc) {
    this.engine = null
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
          `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
          TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
          Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
          ActivationGivenTerm(i, j, k, PROB) :- Activation(i, j, k) // TermAssociation("emotion")
          ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)`,
          `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
          TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
          TermsToSelect("emotion")
          TermsToSelect("fear")
          Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
          ActivationGivenTerm(i, j, k, PROB) :- Activation(i, j, k) // (TermAssociation(t) & TermsToSelect(t))
          ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)`
        ]
      },
      {
        engine: 'destrieux',
        queries: [
          'union(region_union(r)) :- destrieux(..., r)'
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
    console.log(routeParams)
    if ('engine' in routeParams) {
      this.engine = routeParams.engine
    } else {
      this.engine = null
    }
    console.log(this.ready)
    if (this.ready) {
      this.onRouteEngineChange()
    }
  }

  onRouteEngineChange () {
    let selectedEngine = this.engines.find(e => e.engine === this.engine)
    if (!selectedEngine) {
      selectedEngine = this.engines[0]
    }
    this.qc.setRouteEngine(selectedEngine)
  }

  initEnginesMenu () {
    const menu = $('.nl-dataset-dropdown')
    menu.empty()
    for (const engine of this.engines) {
      const item = $(`<a class="item" href="/${engine.engine}" data-link></a>`)
      if ('queries' in engine && engine.queries.length > 1) {
        $('<i class="dropdown icon"></i>').appendTo(item)
        item.text(engine.engine.charAt(0).toUpperCase() + engine.engine.slice(1))
        const submenu = $('<div class="menu"></div>')
        engine.queries.forEach((q, i) => {
          $(`<a class="item" href="/${engine.engine}/${i}" data-link>Example ${i}</a>`).appendTo(submenu)
        })
        submenu.appendTo(item)
      } else {
        item.text(engine.engine.charAt(0).toUpperCase() + engine.engine.slice(1))
      }
      menu.append(item)
    }

    // show dropdown on hover
    $('.main.menu  .ui.dropdown').dropdown({
      on: 'hover'
    })
  }
}
