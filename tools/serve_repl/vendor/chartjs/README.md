# Chart.js

`chart.umd.min.js` is [Chart.js](https://www.chartjs.org) 4.6.1, the UMD
build of its npm package (`npm pack chart.js@4.6.1`, `dist/chart.umd.min.js`),
under the MIT License (`LICENSE.md`).

The activity page of `neurolang-repl-serve` (`tools/serve_repl/activity.html`,
`/admin/activity`) draws its charts with it. The server serves it to the
admins at `/admin/vendor/chart.umd.min.js`, so the page needs nothing from
another site.

To update it: `npm pack chart.js@4`, then copy `package/dist/chart.umd.min.js`
and `package/LICENSE.md` here, and change the version above.
