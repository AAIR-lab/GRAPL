#ifndef STRXML_H
#define STRXML_H

#include <vector>
#include <string>
#include <map>
#include <stack>

class XMLNode
{
 public:
  int type;

  virtual ~XMLNode() { }
  virtual XMLNode* getChild( int i );
  virtual XMLNode* getChild( std::string s );
  virtual int size( void );
  virtual void print( std::ostream &os ) = 0;
  virtual std::string getText( void ) = 0;
  virtual std::string getName( void ) = 0;
  virtual std::string getParam( std::string s ) = 0;
};

typedef XMLNode* XMLNodePtr;
typedef std::pair<std::string,std::string> str_pair;
typedef std::vector<str_pair> str_pair_vec;
typedef std::vector<std::string> str_vec;
typedef std::map<std::string,std::string> str_str_map;
typedef std::vector<XMLNodePtr> node_vec;
typedef std::stack<XMLNodePtr> node_stk;

class XMLText : public XMLNode
{
 public:
  std::string text;

  XMLText();
  virtual ~XMLText() { }
  virtual void print( std::ostream &os );
  virtual std::string getText( void );
  virtual std::string getName( void );
  virtual std::string getParam( std::string s );
};

class XMLParent : public XMLNode
{
 public:
  std::string name;
  str_str_map params;
  node_vec children;

  XMLParent();
  virtual ~XMLParent();
  virtual void print( std::ostream &os );
  XMLNodePtr getChild( int i );
  XMLNodePtr getChild( std::string s );
  virtual int size( void );
  virtual std::string getText( void );
  virtual std::string getName( void );
  virtual std::string getParam( std::string s );
};

class PSink
{
  node_stk s;
 public:
  int error;
  XMLNode *top;

  PSink() { top = 0; error = 0; }
  virtual ~PSink() { }
  virtual void pushNode( std::string name, str_pair_vec params );
  virtual void popNode( std::string name );
  virtual void pushText( std::string text );
  virtual void formaterror( void ) { error = 1; }
  virtual void streamerror( void ) { error = 2; }
};

int dissectNode( XMLNodePtr p, std::string child, std::string &destination );
int parseStream( std::istream &is, PSink &ps );
XMLNodePtr getNodeFromStream( std::istream &is );

std::ostream& operator<<( std::ostream &os, XMLNodePtr &xn );
std::istream& operator>>( std::istream &is, XMLNodePtr &xn );

#endif // STRXML_H
