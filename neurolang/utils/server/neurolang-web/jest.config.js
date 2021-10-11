module.exports = {
  transform: {
    '^.+\\.js$': 'babel-jest'
  },
  moduleNameMapper: {
    '\\.(css|less|scss)$': 'identity-obj-proxy'
  },
  setupFiles: ['jest-canvas-mock'],
  testEnvironment: 'jsdom'
}
