GRAMMAR = u'''
    @@grammar::Datalog
    @@parseinfo :: True
    @@whitespace :: /[\t ]+/

    start = expressions $ ;

    expressions = ( newline ).{ expression };


    probabilistic_expression = (number | int_ext_identifier ) '::' expression ;
    expression = fact | rule | constraint ;
    fact = constant_predicate ;
    rule = head implication body ;
    constraint = body right_implication head ;
    head = head_predicate ;
    body = ( conjunction ).{ predicate } ;

    conjunction = ',' | '&' | '\N{LOGICAL AND}' ;
    implication = ':-' | '\N{LEFTWARDS ARROW}' ;
    right_implication = '-:' | '\N{RIGHTWARDS ARROW}' ;
    head_predicate = identifier'(' [ arguments ] ')' ;
    predicate = int_ext_identifier'(' [ arguments ] ')'
              | negated_predicate
              | logical_constant ;

    constant_predicate = identifier'(' ','.{ literal } ')' ;

    negated_predicate = ('~' | '\u00AC' ) predicate ;

    arguments = ','.{ argument }+ ;
    argument = int_ext_identifier
             | literal
             | function_application ;

    int_ext_identifier = identifier | ext_identifier ;
    ext_identifier = '@'identifier;

    function_application = int_ext_identifier'(' [ arguments ] ')'
                         | arithmetic_operation ;

    arithmetic_operation = term
                         | ('+' | '-' ) term
                         | ('+' | '-' )<{ term }+ ;

    term = factor
         | ('*' | '/' )<{ factor }+ ;
    factor = argument
           | argument '^' argument
           | '(' argument ')' ;

    literal = text
            | number
            | ext_identifier ;

    identifier = /[a-zA-Z][a-zA-Z0-9]*/ ;

    text = '"' /[a-zA-Z0-9]*/ '"' ;
    number = /\\d+/['.'/\\d+/] ;
    logical_constant = TRUE | FALSE ;
    TRUE = 'True' | '\u22A4' ;
    FALSE = 'False' | '\u22A5' ;

    newline = {['\\u000C'] ['\\r'] '\\n'}+ ;
'''
