#/usr/bin/env bash

mandatory_binaries=( "java" "javac" )

for mandatory_binary in "${mandatory_binaries[@]}"
do
  if ! type "$mandatory_binary" > /dev/null; then
    echo "Please install $mandatory_binary"
    exit
  fi
done

if [ "$#" -ne "3" ]; then
  script_name=`basename "$0"`
  cat << EOF

  USAGE
    $script_name GRAMMAR INPUT RULE

  PARAMETERS
    GRAMMAR    the name of the ANTLR grammar
    INPUT      either the name of the file to parse, or the (string)
               source for the parser to process
    RULE       the name of the parser rule to invoke

  EXAMPLE
    $script_name Expr.g4 "(1 + 2) / Pi" parse

EOF
  exit 1
fi

if [ ! -f "$1" ]; then
  echo "no such grammar: $1"
  exit
fi

function get_script_path
{
  pushd `dirname $0` > /dev/null
  local path=`pwd`
  popd > /dev/null
  echo "$path"
}

function check_antlr_jar
{
  if [ ! -f "$1" ]; then
    echo "No ANTLR JAR found, downloading it now..."
    curl -o "$1" "http://www.antlr.org/download/antlr-$antlr_version-complete.jar"
  fi
}

function write_main_class
{
  cat >"$1" <<EOL
  import org.antlr.v4.runtime.CharStreams;
  import org.antlr.v4.runtime.CommonTokenStream;
  import org.antlr.v4.runtime.ParserRuleContext;
  import org.antlr.v4.runtime.Token;
  import java.io.File;
  import java.io.IOException;

  public class Main {

      private static void printPrettyLispTree(String tree) {

          int indentation = 1;

          for (char c : tree.toCharArray()) {

              if (c == '(') {
                  if (indentation > 1) {
                      System.out.println();
                  }
                  for (int i = 0; i < indentation; i++) {
                      System.out.print("  ");
                  }
                  indentation++;
              }
              else if (c == ')') {
                  indentation--;
              }

              System.out.print(c);
          }

          System.out.println();
      }

      public static void main(String[] args) throws IOException {

          String source = "$3";

          ${2}Lexer lexer = new File(source).exists() ?
                  new ${2}Lexer(CharStreams.fromFileName(source)) :
                  new ${2}Lexer(CharStreams.fromString(source));

          CommonTokenStream tokens = new CommonTokenStream(lexer);
          tokens.fill();

          System.out.println("\n[TOKENS]");

          for (Token t : tokens.getTokens()) {

              String symbolicName = ${2}Lexer.VOCABULARY.getSymbolicName(t.getType());
              String literalName = ${2}Lexer.VOCABULARY.getLiteralName(t.getType());

              System.out.printf("  %-20s '%s'\n",
                      symbolicName == null ? literalName : symbolicName,
                      t.getText().replace("\r", "\\r").replace("\n", "\\n").replace("\t", "\\t"));
          }

          System.out.println("\n[PARSE-TREE]");

          ${2}Parser parser = new ${2}Parser(tokens);
          ParserRuleContext context = parser.${4}();

          String tree = context.toStringTree(parser);
          printPrettyLispTree(tree);
      }
  }
EOL
}

# Declare some variables
grammar_file="$1"
input="$2"
rule_name="$3"
main_class_name="Main"
main_class_file="$main_class_name.java"
grammar_name=${grammar_file%.*}
antlr_version="4.7.1"
script_path=$(get_script_path)
antlr_jar="$script_path/antlr-$antlr_version-complete.jar"

# Make sure the ANTLR jar is available
#check_antlr_jar "$antlr_jar"

antlr_jar=$ANTLR_PATH

# Generate the lexer and parser classes
java -cp "$antlr_jar" org.antlr.v4.Tool "$grammar_file"

# Generate a main class
write_main_class "$main_class_file" "$grammar_name" "$input" "$rule_name"

# Compile all .java source files and run the main class
javac -cp "$antlr_jar:." *.java
java -cp "$antlr_jar:." "$main_class_name"