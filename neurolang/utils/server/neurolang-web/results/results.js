import $ from "jquery"
import "./results.css"


export function showQueryResults(queryId, results) {
    resultsContainer.show()
    createResultTabs(results.data.results)
}

const resultsContainer = $("#resultsContainer")
const resultsTabs = resultsContainer.find(".nl-results-tabs")
const tabTable = resultsContainer.find(".nl-result-table")

function createResultTabs(results, defaultTab) {
    // clear tabs
    resultsTabs.empty()

    // add new tabs
    for (const symbol in results) {
        const tab = $(`<li class='nav-item nav-link'>${symbol}</li>`)
        if (typeof defaultTab === 'undefined' || defaultTab === symbol) {
            tab.addClass("active")
            defaultTab = symbol
        }
        tab.on("click", (evt) => setActiveResultTab(evt, results, symbol))
        resultsTabs.append(tab)
    }

    // display selected tab
    setActiveResultTab(null, results, defaultTab)
}

function setActiveResultTab(evt, results, symbol) {
    if (evt !== null) {
        resultsContainer.find("li.active").removeClass("active")
        $(evt.target).addClass("active")
    }

    // clear previous tab results
    if ($.fn.DataTable.isDataTable(tabTable)) {
        tabTable.DataTable().destroy()
        tabTable.empty();
    }

    // set new data
    const tab = results[symbol]
    const cols = tab.columns.map(col => ({ title: col }))
    tabTable.DataTable({
        data: tab.values,
        columns: cols
    });
}

function clearResultsContainer() {
    resultsContainer.display = "none"
}