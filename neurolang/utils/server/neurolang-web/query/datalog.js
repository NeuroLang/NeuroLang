/**
 * Datalog CodeMirror mode taken from
 * https://github.com/ysangkok/mitre-datalog.js/blob/master/datalogCodeMirrorMode.js
 */

import CodeMirror from 'codemirror'

CodeMirror.defineMode('datalog', function (cmCfg, modeCfg) {
  function rval (state, stream, type) {
    state.context = false

    // remember last significant bit on last line for indenting
    if (type !== 'whitespace' && type !== 'comment') {
      state.lastToken = stream.current()
    }
    //     erlang             :- CodeMirror tag
    switch (type) {
      case 'atom': return 'atom'
      case 'attribute': return 'attribute'
      case 'builtin': return 'builtin' // ~
      case 'comment': return 'comment' // %
        //      case "fun":         return "meta";
      case 'function': return 'tag' // a in a(x).
        //      case "guard":       return "property";
      case 'keyword': return 'keyword' // ?
        //      case "macro":       return "variable-2";
      case 'number': return 'number'
      case 'operator': return 'operator' // =
        //      case "record":      return "bracket";
      case 'string': return 'string'
        //     case "type":        return "def";
      case 'variable': return 'variable'
      case 'error': return 'error'
      case 'separator': return 'variable-2' // :-
      case 'open_paren': return null
      case 'close_paren': return null
      case 'whitespace': return null
      default: return null
    }
  }

  const separatorWords = [
    ':-', ':', '.', ',']

  const symbolWords = [
    '=']

  const openParenWords = [
    '(']

  const smallRE = /[a-z_]/
  const largeRE = /[A-Z_]/
  const digitRE = /[0-9]/
  const anumRE = /[a-z_A-Z0-9]/
  const symbolRE = /[=]/
  const openParenRE = /[\(]/
  const sepRE = /[\-\.,:]/

  function isMember (element, list) {
    return (list.indexOf(element) > -1)
  }

  function isPrev (stream, string) {
    const start = stream.start
    const len = string.length
    if (len <= start) {
      const word = stream.string.slice(start - len, start)
      return word === string
    } else {
      return false
    }
  }

  function tokenize (stream, state) {
    if (stream.eatSpace()) {
      return rval(state, stream, 'whitespace')
    }

    const ch = stream.next()

    // comment
    if (ch === '%') {
      stream.skipToEnd()
      return rval(state, stream, 'comment')
    }

    // string
    if (ch === '"') {
      if (doubleQuote(stream)) {
        return rval(state, stream, 'string')
      } else {
        return rval(state, stream, 'error')
      }
    }

    // variable
    if (largeRE.test(ch)) {
      stream.eatWhile(anumRE)
      return rval(state, stream, 'variable')
    }

    // atom/keyword/BIF/function
    if (smallRE.test(ch)) {
      stream.eatWhile(anumRE)

      const w = stream.current()

      if (stream.peek() === '(') {
        return rval(state, stream, 'function')
      }
      return rval(state, stream, 'atom')
    }

    // number
    if (digitRE.test(ch)) {
      stream.eatWhile(digitRE)
      return rval(state, stream, 'number') // normal integer
    }

    // open parens
    if (nongreedy(stream, openParenRE, openParenWords)) {
      pushToken(state, stream)
      return rval(state, stream, 'open_paren')
    }

    // close parens
    if (nongreedy(stream, /\)/, [')'])) {
      pushToken(state, stream)
      return rval(state, stream, 'close_paren')
    }

    // separators
    if (greedy(stream, sepRE, separatorWords)) {
      // distinguish between "." as terminator and record field operator
      if (state.context === false) {
        pushToken(state, stream)
      }
      return rval(state, stream, 'separator')
    }

    // operators
    if (greedy(stream, symbolRE, symbolWords)) {
      return rval(state, stream, 'operator')
    }

    if (greedy(stream, /\?/, ['?'])) {
      return rval(state, stream, 'keyword')
    }

    if (greedy(stream, /\~/, ['~'])) {
      return rval(state, stream, 'builtin')
    }

    return rval(state, stream, null)
  }

  function nongreedy (stream, re, words) {
    if (stream.current().length === 1 && re.test(stream.current())) {
      stream.backUp(1)
      while (re.test(stream.peek())) {
        stream.next()
        if (isMember(stream.current(), words)) {
          return true
        }
      }
      stream.backUp(stream.current().length - 1)
    }
    return false
  }

  function greedy (stream, re, words) {
    if (stream.current().length === 1 && re.test(stream.current())) {
      while (re.test(stream.peek())) {
        stream.next()
      }
      while (stream.current().length > 0) {
        if (isMember(stream.current(), words)) {
          return true
        } else {
          stream.backUp(1)
        }
      }
      stream.next()
    }
    return false
  }

  function doubleQuote (stream) {
    return quote(stream, '"', '\\')
  }

  function quote (stream, quoteChar, escapeChar) {
    while (!stream.eol()) {
      const ch = stream.next()
      if (ch === quoteChar) {
        return true
      } else if (ch === escapeChar) {
        stream.next()
      }
    }
    return false
  }

  function Token (stream) {
    this.token = stream ? stream.current() : ''
    this.column = stream ? stream.column() : 0
    this.indent = stream ? stream.indentation() : 0
  }

  function myIndent (state, textAfter) {
    const indent = cmCfg.indentUnit
    const token = (peekToken(state)).token
    const wordAfter = takewhile(textAfter, /[^a-z]/)

    if (isMember(token, openParenWords)) {
      return (peekToken(state)).column + token.length
    } else if (token === '.' || token === '') {
      return 0
    } else if (token === ':-') {
      return (peekToken(state)).indent + indent
    } else {
      return (peekToken(state)).column + indent
    }
  }

  function takewhile (str, re) {
    const m = str.match(re)
    return m ? str.slice(0, m.index) : str
  }

  function popToken (state) {
    return state.tokenStack.pop()
  }

  function peekToken (state, depth) {
    const len = state.tokenStack.length
    const dep = (depth || 1)
    if (len < dep) {
      return new Token()
    } else {
      return state.tokenStack[len - dep]
    }
  }

  function pushToken (state, stream) {
    const token = stream.current()
    const prev_token = peekToken(state).token
    if (token === ',') {
      return false
    } else if (drop_both(prev_token, token)) {
      popToken(state)
      return false
    } else if (drop_first(prev_token, token)) {
      popToken(state)
      return pushToken(state, stream)
    } else {
      state.tokenStack.push(new Token(stream))
      return true
    }
  }

  function drop_first (open, close) {
    switch (open + ' ' + close) {
      case ':- .': return true
      default: return false
    }
  }

  function drop_both (open, close) {
    switch (open + ' ' + close) {
      case '( )': return true
      default: return false
    }
  }

  return {
    startState:
         function () {
           return {
             tokenStack: [],
             context: false,
             lastToken: null
           }
         },

    token:
         function (stream, state) {
           return tokenize(stream, state)
         },

    indent:
         function (state, textAfter) {
           //        console.log(state.tokenStack);
           return myIndent(state, textAfter)
         }
  }
})
