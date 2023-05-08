# A class to handle drawing and updating the GUI for the human player

import pygame
import sys
from Board_opperations import check_jump_required, generate_options, update_board


class Gui(): # class to deal with the visual elements for the human player
    C1 = (186,45,30)
    C2 = (0,0,0)
    C3 = (126,25,10)
    C4 = (40,40,40)
    CB = (179,149,96)
    CD = (125,107,79)
    CP = (227,116,52)
    CS = (167,229,211)
    CM = (174,181,161)
    CR = (174,141,121)
    CL = (174,181,181)
    CZ = (154,161,161)

    def __init__(self, board: list, size : tuple, clock : object, screen : object, type_ : int) -> None:
        self.type = type_
        if self.type == 1:
            self.king_type = 3
        else:
            self.king_type = 4
        self.board = board
        self.selected_block = None
        self.highlighted_blocks = []
        self.limited_options = []
        self.red_blocks = []
        self.blue_blocks = []
        self.size = size
        self.clock = clock
        self.screen = screen

        # data displayed on the side bar
        self.leafs = 0
        self.hashes = 0
        self.montycarlop1 = 0
        self.montycarlop2 = 0
        self.depth = 0
        self.eval = 0
        self.win_messsage = ""

    # the main loop that runs when the human player is choosing its next move (yes it is a horribly complicated function)
    def choose_action(self) -> tuple:
        move = None
        clock = pygame.time.Clock()
        self.limited_options = check_jump_required(self.board, self.type)
        if len(self.limited_options) == 1:
            self.selected_block = self.limited_options[0]
            self.highlighted_blocks = generate_options((self.selected_block[0],self.selected_block[1]), self.board, only_jump=True)
        running = True
        while running:
            mouse_button_click = []
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                    pygame.quit()
                    sys.exit()
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    mouse_button_click.append(event.button)
            # GUI loop starts here
            if mouse_button_click: # handle a click
                mouse_x, mouse_y = pygame.mouse.get_pos()
                for y, row in enumerate(self.board):
                    for x, spot in enumerate(row):
                        # check if the player tryed to move
                        for block in self.highlighted_blocks:
                            if self.size[1] / 8 * x < mouse_x < self.size[1] / 8 * (x + 1)\
                                and self.size[1] / 8 * y < mouse_y < self.size[1] / 8 * (y + 1):
                                if (x, y) == block:
                                    move = (tuple(self.selected_block),(x, y))
                                    hopped = update_board(tuple(self.selected_block), (x, y), self.board)
                                    self.selected_block = None
                                    self.blue_blocks = move
                                    self.highlighted_blocks = []
                                    self.limited_options = []
                                    self.red_blocks = []
                                    if hopped:
                                        if not generate_options((x,y), self.board, only_jump=True) == []:
                                            return True, move
                                        else:
                                            running = False
                                            break
                                    else:
                                        running = False
                                        break
                        # check if the player selected a block
                        if self.size[1] / 8 * x < mouse_x < self.size[1] / 8 * (x + 1)\
                             and self.size[1] / 8 * y < mouse_y < self.size[1] / 8 * (y + 1)\
                                 and (spot == self.type or spot == self.king_type):
                            if self.limited_options == []:
                                self.selected_block = (x, y)
                                self.highlighted_blocks = generate_options((x,y), self.board)
                            else:
                                for i in self.limited_options:
                                    if i == (x, y):
                                        self.selected_block = (x, y)
                                        self.highlighted_blocks = generate_options((x,y), self.board, only_jump=True)
            self.draw()
            clock.tick(60)
        return False, move

    # draws the board
    def draw(self) -> None:
        # draw the blocks
        mouse_x, mouse_y = pygame.mouse.get_pos()
        block_width = self.size[1]/8
        block_height = self.size[1]/8
        color = lambda x : Gui.CB if x % 2 == 0 else Gui.CD 
        for y, row in enumerate(self.board):
            for x, spot in enumerate(row):
                i = x + y
                c = color(i)
                rect = (block_width * x,
                        block_height * y,
                        block_width * (x + 1),
                        block_height * (y + 1))
                for piece in self.red_blocks:
                    if (x, y) == piece:
                        c = Gui.CR
                for piece in self.blue_blocks:
                    if (x, y) == piece:
                        c = Gui.CZ
                for piece in self.highlighted_blocks:
                    if (x, y) == piece:
                        c = Gui.CM
                for piece in self.limited_options:
                    if (x, y) == piece:
                        c = Gui.CL
                if (x, y) == self.selected_block:
                    c = Gui.CS
                if rect[0] < mouse_x < rect[2] and rect[1] < mouse_y < rect[3]:
                    c = (c[0] + 15, c[1] + 15, c[2] + 15)
                pygame.draw.rect(self.screen, c, rect)
                
        # draw the pieces
        for y, row in enumerate(self.board):
            for x, spot in enumerate(row):
                spot_on_screen = (x * block_width + block_width * .5,
                    y * block_height + block_height * .5)
                if self.board[y][x] == 1 or self.board[y][x] == 3:
                    pygame.draw.circle(self.screen, Gui.C1, spot_on_screen, block_width / 2 - 10)
                    if self.board[y][x] == 3:
                        pygame.draw.circle(self.screen, Gui.C3, spot_on_screen, block_width / 2 - 20)
                elif self.board[y][x] == 2 or self.board[y][x] == 4:
                    pygame.draw.circle(self.screen, Gui.C2, spot_on_screen, block_width / 2 - 10)
                    if self.board[y][x] == 4:
                        pygame.draw.circle(self.screen, Gui.C4, spot_on_screen, block_width / 2 - 20)

        # draw the data for the side bar
        pygame.draw.rect(self.screen, Gui.CD, (block_width * 8, 0, self.size[0], self.size[1]))
        pygame.draw.line(self.screen, Gui.C2, (block_width * 8, 0), (block_width * 8, self.size[1]), 4)
        # draw the leafs searched
        self.screen.blit(pygame.font.SysFont('Corbel', 18).render('Nodes:', True, (0, 0, 0)),
                    (self.size[1] + 10, self.size[1]/8 * 1))
        self.screen.blit(pygame.font.SysFont('Corbel', 16).render(str(self.leafs), True, (0, 0, 0)),
                    (self.size[1] + 20, self.size[1]/8 * 1.5))
        # draw the hashes generated
        self.screen.blit(pygame.font.SysFont('Corbel', 18).render('Transpositions:', True, (0, 0, 0)),
            (self.size[1] + 10, self.size[1]/8 * 2))
        self.screen.blit(pygame.font.SysFont('Corbel', 16).render(str(self.hashes), True, (0, 0, 0)),
                    (self.size[1] + 20, self.size[1]/8 * 2.5))

        # draw the depth
        self.screen.blit(pygame.font.SysFont('Corbel', 18).render('Depth Reached:', True, (0, 0, 0)),
            (self.size[1] + 10, self.size[1]/8 * 3))
        self.screen.blit(pygame.font.SysFont('Corbel', 16).render(str(self.depth), True, (0, 0, 0)),
                    (self.size[1] + 20, self.size[1]/8 * 3.5))
        # draw the eval found by minimax
        self.screen.blit(pygame.font.SysFont('Corbel', 18).render('Board Eval:', True, (0, 0, 0)),
            (self.size[1] + 10, self.size[1]/8 * 4))
        self.screen.blit(pygame.font.SysFont('Corbel', 16).render(str(round(self.eval, 4)), True, (0, 0, 0)),
                    (self.size[1] + 20, self.size[1]/8 * 4.5))
        # win message
        self.screen.blit(pygame.font.SysFont('Corbel', 20).render(self.win_messsage, True, (0, 0, 0)),
                    (self.size[1] + 20, self.size[1]/8 * 5.5))

        pygame.display.update()

    def update_params(self, data):
        self.leafs = data["leafs"]
        self.depth = data["depth"]
        self.hashes = data["hashes"]
        self.eval = data["eval"]