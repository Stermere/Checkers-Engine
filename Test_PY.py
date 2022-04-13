import search_engine

def main():
    p1 = 6172839697753047040;
    p2 = 11163050;
    p1k = 0;
    p2k = 0;

    player = 1

    return_data = search_engine.search_position(p1, p2, p1k, p2k, player)

    print(return_data)

if __name__ == "__main__":
    main()
