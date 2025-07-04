CUSTOMFLAGS=-Werror -Wall -Wno-unused-result -std=gnu99 -g -O1
SANITIZE=-fsanitize=address -fsanitize=leak -fsanitize=undefined -fsanitize=unreachable -fsanitize=null -fsanitize=bounds-strict 
BINARY=exec
MAIN=main.c
UTIL=record.c utilities.c import.c show.c parser.c find.c manipulation.c header.c querry.c
ZIPNAME=entrega.zip 

all:
	gcc $(CUSTOMFLAGS) $(UTIL) $(MAIN) -o $(BINARY) -lm

sanitize:
	gcc $(CUSTOMFLAGS) $(UTIL) $(MAIN) $(SANITIZE) -o $(BINARY) -lm 

run:
	./$(BINARY)

valgrind:
	valgrind --tool=memcheck --leak-check=full	--track-origins=yes --show-leak-kinds=all --show-reachable=yes --vgdb-error=1 ./$(BINARY)

debug:
	gcc *.c -o exec $(CUSTOMFLAGS) -DDEBUG

zip:
	zip entrega.zip *.c *.h Makefile

clean:
	@rm -f $(ZIPNAME) $(BINARY) 
