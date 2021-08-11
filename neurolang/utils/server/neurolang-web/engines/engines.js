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
    // $.get(API_ROUTE.engines)
    //   .done((data) => {
    //     this.engines = data
    //   })
    this.engines = [
      {
        engine: 'neurosynth',
        queries: [
          {
            title: 'Coordinate-based meta-analysis (CBMA) on the Neurosynth database',
            shortTitle: 'CBMA Single Term',
            id: 'neuro1',
            query: `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
ActivationGivenTerm(i, j, k, PROB) :- Activation(i, j, k) // TermAssociation("emotion")
ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)`,
            description: `In this example, we use the Neurosynth CBMA database (Yarkoni et al., 2011), consisting of 14,371 studies.
We load the data from the Neurosynth database into tables in the Neurolang engine :
* **PeakReported** is a relation, or tuple, that includes the peak coordinates (i, j, k) reported in each study.
* **Study** is a relation, or tuple, with one variable corresponding to the *id* of each study.
* **SelectedStudy** annotates each study with a probability equal to 1/N of it being chosen out of the whole dataset of size N.

We write a probabilistic program to query the probability of a peak coordinate being reported by a study given that
this study mentions a specific term (i.e. **emotion**).`
          },
          {
            title: 'Coordinate-based meta-analysis (CBMA) on the Neurosynth database',
            shortTitle: 'CBMA Multiple Terms',
            id: 'neuro2',
            query: `TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
TermsToSelect("emotion")
TermsToSelect("fear")
Activation(i, j, k) :- SelectedStudy(s) & PeakReported(i, j, k, s)
ActivationGivenTerm(i, j, k, t, PROB) :- Activation(i, j, k) // (TermAssociation(t) & TermsToSelect(t))
ActivationGivenTermImage(t, agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, t, p)`,
            description: `This example is similar to the [CBMA Single Term](/neurosynth/neuro1) example but it showcases
how to query multiple term associations.`
          },
          {
            title: 'Coordinate-based meta-analysis (CBMA) with spatial prior smoothing',
            shortTitle: 'CBMA Spatial Prior',
            id: 'neuro3',
            query: `VoxelReported (i, j, k, study) :- PeakReported(i2, j2, k2, study) & Voxel(i, j, k) & (d == EUCLIDEAN(i, j, k, i2, j2, k2)) & (d < 1)
TermInStudy(term, study) :: tfidf :- TermInStudyTFIDF(term, tfidf, study)
TermAssociation(term) :- SelectedStudy(study) & TermInStudy(term, study)
Activation(i, j, k) :- SelectedStudy(s) & VoxelReported(i, j, k, s)
ActivationGivenTerm(i, j, k, PROB) :- Activation(i, j, k) // TermAssociation("emotion")
ActivationGivenTermImage(agg_create_region_overlay(i, j, k, p)) :- ActivationGivenTerm(i, j, k, p)`,
            description: `This example illustrates how a spatial prior can be defined based on the distance between voxels
and foci in a coordinate-based meta-analysis database.
            
Here, each voxel's probability of being reported by a study is calculated based on whether that particular study
reports a focus (peak activation) near the voxel. The probability is defined based on how far from the focus that
voxel happens to be.`
          }
        ]
      },
      {
        engine: 'destrieux',
        queries: [
          {
            title: 'Union of Destrieux atlas regions',
            shortTitle: 'Union of atlas regions',
            id: 'destrieux1',
            query: 'union(region_union(r)) :- destrieux(..., r)'
          },
          {
            title: 'Sulcal Identification Query Example in Neurolang',
            shortTitle: 'Sulcal identification',
            id: 'destrieux2',
            query: `LeftSulcus(name_, region) :- destrieux(name_, region) & startswith("L S", name_)
LeftPrimarySulcusName("L S central")
LeftPrimarySulcusName("L Lat Fis post")
LeftPrimarySulcusName("L S pericallosal")
LeftPrimarySulcusName("L S parieto occipital")
LeftPrimarySulcusName("L S calcarine")
LeftPrimarySulcusName("L Lat Fis ant Vertical")
LeftPrimarySulcusName("L Lat Fis ant Horizont")
LeftPrimarySulcus(name_, region) :- destrieux(name_, region) & LeftPrimarySulcusName(name_)
LeftFrontalLobeSulcus(region) :- LeftSulcus(..., region) & anatomical_anterior_of(region, lscregion) & destrieux("L S central", lscregion) & anatomical_superior_of(region, llfavregion) & destrieux("L Lat Fis ant Vertical", llfavregion)
LPrecentralSulcus(r) :- LeftFrontalLobeSulcus(r) & principal_direction(r, "SI") & ~exists(r2; (LeftFrontalLobeSulcus(r2) & (r2 != r) & anatomical_posterior_of(r2, r)))`,
            description: `In this example, we first caracterise some of the sulci in the Destrieux et al. Atlas.
We characterise:
* the left hemisphere primary sulci, by name
* the left frontal lobe sulcus as those
    - anterior to Destrieux's left central sulcus
    - superior to Destrieux's left anterio-vertical section of the lateral fissure.
    
We then identify the left precentral sulcus (PC) as:
* belonging to the left frontal lobe
* its principal direction is along the superior-inferior axis.
* no other sulcus satisfying the same conditions is anterior to the PC.`
          }
        ]
      }
    ]
    this.initEnginesMenu()
    const queryAccordion = $('.nl-query-info')
    queryAccordion.accordion()
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
