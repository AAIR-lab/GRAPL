%{
#ifdef YYDEBUG
  extern int yydebug=1;
#endif


#include <stdio.h>
#include <string.h> 
#include "ff.h"
#include "memory.h"
#include "parse.h"


#ifndef SCAN_ERR
#define SCAN_ERR
#define DEFINE_EXPECTED            0
#define PROBLEM_EXPECTED           1
#define PROBNAME_EXPECTED          2
#define LBRACKET_EXPECTED          3
#define RBRACKET_EXPECTED          4
#define DOMDEFS_EXPECTED           5
#define REQUIREM_EXPECTED          6
#define TYPEDLIST_EXPECTED         7
#define DOMEXT_EXPECTED            8
#define DOMEXTNAME_EXPECTED        9
#define TYPEDEF_EXPECTED          10
#define CONSTLIST_EXPECTED        11
#define PREDDEF_EXPECTED          12 
#define NAME_EXPECTED             13
#define VARIABLE_EXPECTED         14
#define ACTIONFUNCTOR_EXPECTED    15
#define ATOM_FORMULA_EXPECTED     16
#define EFFECT_DEF_EXPECTED       17
#define NEG_FORMULA_EXPECTED      18
#define NOT_SUPPORTED             19
#define SITUATION_EXPECTED        20
#define SITNAME_EXPECTED          21
#define BDOMAIN_EXPECTED          22
#define BADDOMAIN                 23
#define INIFACTS                  24
#define PLANDEF                   25
#define ADLGOAL                   26
#endif


static char * serrmsg[] = {
  "'define' expected",
  "'problem' expected",
  "problem name expected",
  "'(' expected",
  "')' expected",
  "additional domain definitions expected",
  "requirements (e.g. ':strips') expected",
  "typed list of <%s> expected",
  "domain extension expected",
  "domain to be extented expected",
  "type definition expected",
  "list of constants expected",
  "predicate definition expected",
  "<name> expected",
  "<variable> expected",
  "action functor expected",
  "atomic formula expected",
  "effect definition expected",
  "negated atomic formula expected",
  "requirement %s not supported by this IPP version",  
  "'situation' expected",
  "situation name expected",
  "':domain' expected",
  "this problem needs another domain file",
  "initial facts definition expected",
  "plan definition expected",
  "first order logic expression expected",
  NULL
};


/* void planerr( int errno, char *par ); */


static int sact_err;
static char *sact_err_par = NULL;
static Bool sis_negated = FALSE;

%}


%start file


%union {

  char string[MAX_LENGTH];
  char* pstring;
  PlNode* pPlNode;
  FactList* pFactList;
  TokenList* pTokenList;
  TypedList* pTypedList;

}


%type <pstring> problem_name
%type <pPlNode> literal_name_plus
%type <pPlNode> literal_name
%type <pTokenList> name_star
%type <pTokenList> atomic_formula_name
%type <pstring> predicate
%type <pTypedList> typed_list_name
%type <pTokenList> name_plus
%type <pTokenList> action_list
%type <pstring> action_name
%type <pPlNode> action_literal_name
%type <pTokenList> action_formula_name

%token DEFINE_TOK
%token PROBLEM_TOK
%token SITUATION_TOK
%token BSITUATION_TOK
%token OBJECTS_TOK
%token BDOMAIN_TOK
%token INIT_TOK
%token PLAN_TOK
%token EQ_TOK
%token AND_TOK
%token NOT_TOK
%token <string> NAME
%token <string> VARIABLE
%token <string> TYPE
%token FORALL_TOK
%token IMPLY_TOK
%token OR_TOK
%token EXISTS_TOK
%token EITHER_TOK
%token OPEN_PAREN
%token CLOSE_PAREN

%%


/**********************************************************************/
file:
/* empty */
|
problem_definition  file
;


/**********************************************************************/
problem_definition : 
OPEN_PAREN DEFINE_TOK         
{ 
  planerr( PROBNAME_EXPECTED, NULL ); 
}
problem_name  problem_defs  CLOSE_PAREN                 
{  
  gproblem_name = $4;
  if ( gcmd_line.display_info >= 1 ) {
    printf("\nproblem '%s' defined\n", gproblem_name);
  }
}
;


/**********************************************************************/
problem_name :
OPEN_PAREN  PROBLEM_TOK  NAME  CLOSE_PAREN
{
  $$ = new_Token( strlen($3)+1 );
  strcpy($$, $3);
}
;


/**********************************************************************/
base_domain_name :
OPEN_PAREN  BDOMAIN_TOK  NAME  CLOSE_PAREN
{
  if ( SAME != strcmp($3, gdomain_name) ) {
    planerr( BADDOMAIN, NULL );
    yyerror();
  }
}
;


/**********************************************************************/
problem_defs:
/* empty */
|
objects_def  problem_defs
|
init_def  problem_defs
|
plan_def  problem_defs
|
base_domain_name  problem_defs
;


/**********************************************************************/
objects_def:
OPEN_PAREN  OBJECTS_TOK  typed_list_name  CLOSE_PAREN
{
  gparse_objects = $3;
}
;


/**********************************************************************/
init_def:
OPEN_PAREN  INIT_TOK
{
  planerr( INIFACTS, NULL );
}
literal_name_plus  CLOSE_PAREN
{
  gorig_initial_facts = new_PlNode(AND);
  gorig_initial_facts->sons = $4;
}
;


/**********************************************************************/
plan_def:
OPEN_PAREN  PLAN_TOK
{
  planerr( PLANDEF, NULL );
}
action_list  CLOSE_PAREN
{

  gorig_plan_facts = new_PlNode(AND);
  gorig_plan_facts->sons = $4;

}
;

/**********************************************************************/
name_plus:
NAME
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
}
|
NAME  name_plus
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
  $$->next = $2;
}
;

/**********************************************************************/
typed_list_name:     /* returns TypedList */
/* empty */
{ $$ = NULL; }
|
NAME  EITHER_TOK  name_plus  CLOSE_PAREN  typed_list_name
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = $3;
  $$->next = $5;
}
|
NAME  TYPE  typed_list_name   /* end of list for one type */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  $$->type = new_TokenList();
  $$->type->item = new_Token( strlen($2)+1 );
  strcpy( $$->type->item, $2 );
  $$->next = $3;
}
|
NAME  typed_list_name        /* a list element (gets type from next one) */
{
  $$ = new_TypedList();
  $$->name = new_Token( strlen($1)+1 );
  strcpy( $$->name, $1 );
  if ( $2 ) {/* another element (already typed) is following */
    $$->type = copy_TokenList( $2->type );
  } else {/* no further element - it must be an untyped list */
    $$->type = new_TokenList();
    $$->type->item = new_Token( strlen(STANDARD_TYPE)+1 );
    strcpy( $$->type->item, STANDARD_TYPE );
  }
  $$->next = $2;
}
;


/**********************************************************************/
predicate:
NAME
{
  $$ = new_Token(strlen($1) + 1);
  strcpy($$, $1);
}
;


/**********************************************************************/
literal_name_plus:
literal_name
{
  $$ = $1;
}
|
literal_name literal_name_plus
{
   $$ = $1;
   $$->next = $2;
}
;

/**********************************************************************/
literal_name:
OPEN_PAREN  NOT_TOK  atomic_formula_name  CLOSE_PAREN
{
  PlNode *tmp;

  tmp = new_PlNode(ATOM);
  tmp->atom = $3;
  $$ = new_PlNode(NOT);
  $$->sons = tmp;
}
|
atomic_formula_name
{
  $$ = new_PlNode(ATOM);
  $$->atom = $1;
}
;


/**********************************************************************/
atomic_formula_name:
OPEN_PAREN  predicate  name_star  CLOSE_PAREN
{
  $$ = new_TokenList();
  $$->item = $2;
  $$->next = $3;
}
;


/**********************************************************************/
name_star:
/* empty */
{ $$ = NULL; }
|
NAME  name_star
{
  $$ = new_TokenList();
  $$->item = new_Token(strlen($1) + 1);
  strcpy($$->item, $1);
  $$->next = $2;
}
;


/**********************************************************************/
action_name:
NAME
{
  $$ = new_Token(strlen($1) + 1);
  strcpy($$, $1);
}
;
/**********************************************************************/
action_list:
action_literal_name
{
  $$ = $1;
}
|
action_literal_name action_list
{
   $$ = $1;
   $$->next = $2;
}
;

/**********************************************************************/
action_literal_name:
OPEN_PAREN  NOT_TOK action_formula_name  CLOSE_PAREN
{
  PlNode *tmp;

  tmp = new_PlNode(ATOM);
  tmp->atom = $3;
  $$ = new_PlNode(NOT);
  $$->sons = tmp;
}
|
action_formula_name
{
  $$ = new_PlNode(ATOM);
  $$->atom = $1;
}
;

/**********************************************************************/
action_formula_name:
OPEN_PAREN  action_name  name_star  CLOSE_PAREN
{
  $$ = new_TokenList();
  $$->item = $2;
  $$->next = $3;
}
;
/**********************************************************************/
%%


#include "lex.plan_pddl.c"


/**********************************************************************
 * Functions
 **********************************************************************/


/*
 * call	bison -pfct -bscan-fct scan-fct.y
 */
void planerr( int errno, char *par ) {

/*
   sact_err = errno;
   if ( sact_err_par ) {
     free( sact_err_par );
   }*/
/*   if ( par ) { */
/*     sact_err_par = new_Token( strlen(par)+1 ); */
/*     strcpy( sact_err_par, par); */
/*   } else { */
/*     sact_err_par = NULL; */
/*   } */

}



int yyerror( char *msg )

{
  fflush( stdout );
  fprintf(stderr,"\n%s: syntax error in line %d, '%s':\n",
	  gact_filename, lineno, yytext );

  if ( sact_err_par ) {
    fprintf(stderr, "%s%s\n", serrmsg[sact_err], sact_err_par );
  } else {
    fprintf(stderr,"%s\n", serrmsg[sact_err] );
  }

  exit( 1 );

}



void load_plan_file( char *filename )

{

  FILE *fp;/* pointer to input files */
  char tmp[MAX_LENGTH] = "";

  /* open fact file
   */
  if( ( fp = fopen( filename, "r" ) ) == NULL ) {
    sprintf(tmp, "\nff: can't find plan file: %s\n\n", filename );
    perror(tmp);
    exit ( 1 );
  }

  gact_filename = filename;
  lineno = 1;
  yyin = fp;

  yyparse();

  fclose( fp );/* and close file again */

}
