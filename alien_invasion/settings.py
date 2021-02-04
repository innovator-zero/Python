class Settings():
    """存储所有设置的类"""

    def __init__(self):
        """初始化游戏的静态设置"""
        # 屏幕设置
        self.screen_width = 1200
        self.screen_height = 800
        self.bg_color = (230, 230, 230)

        # 子弹设置
        self.bullet_width = 3
        self.bullet_height = 15
        self.bullet_color = 255, 0, 0
        self.bullets_allowed = 3

        self.fleet_drop_speed = 10
        self.ship_limit=3

        #以什么样的速度加快游戏节奏
        self.speedup_scale=1.1
        self.score_scale=1.5

        self.initialize_dynamic_settings()

    def initialize_dynamic_settings(self):
        """初始化随游戏进行变化的设置"""
        self.ship_speed_factor = 1.5
        self.bullet_speed_factor = 3
        self.alien_speed_factor = 1
        # fleet_direction 1=right -1=left
        self.fleet_direction = 1

        #score
        self.alien_points= 50

    def increse_speed(self):
        self.ship_speed_factor*=self.speedup_scale
        self.bullet_speed_factor *= self.speedup_scale
        self.alien_speed_factor *= self.speedup_scale
        self.alien_points=int(self.alien_points*self.score_scale)