import $ from '../jquery-bundler'

const pathToRegex = path => new RegExp('^' + path.replace(/\//g, '\\/').replace(/:\w+/g, '(.+)') + '$')

const getParams = match => {
  const values = match.result.slice(1)
  const keys = Array.from(match.route.path.matchAll(/:(\w+)/g)).map(result => result[1])

  return Object.fromEntries(keys.map((key, i) => {
    return [key, values[i]]
  }))
}

export class Router {
  constructor (routes) {
    this.routes = routes
    this._init()
  }

  /**
   * Initialize the router behaviour. The router listens for
   * 'popstate' changes (i.e back button on navigator) and for click
   * events on links with a [data-link] attribute. When such an event
   * is triggered, the route method is invoked.
   * Also invokes the route method on page load.
   */
  _init () {
    // Listen for changes with `back` / `forward` buttons
    window.addEventListener('popstate', () => this.route())

    // Listen for clicks on links with `data-link` attributes
    document.addEventListener('DOMContentLoaded', () => {
      document.body.addEventListener('click', e => {
        const closest = e.target.closest('[data-link]')
        if (closest) {
          e.preventDefault()
          this.navigateTo(closest.href)
        }
      })

      this.route()
    })
  }

  /**
   * Convenience method to navigate to the given url.
   * This will update the path for the page in the navigator
   * @param {*} url
   */
  navigateTo (url) {
    history.pushState(null, null, url)
    this.route()
  }

  /**
   * Resolve the page's location by trying to match the current path
   * to one of the Router's routes.
   * The Router will then call the `onRouteChange` method for the matched
   * route with the route params.
   */
  async route () {
    // Test each route for potential match given the location path
    const potentialMatches = this.routes.map(route => {
      return {
        route: route,
        result: location.pathname.match(pathToRegex(route.path))
      }
    })

    let match = potentialMatches.find(potentialMatch => potentialMatch.result !== null)
    if (!match) {
      match = {
        route: this.routes[0],
        result: [location.pathname]
      }
    }

    match.route.onRouteChange(getParams(match))
  }
}
