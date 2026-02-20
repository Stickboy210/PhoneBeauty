from pythonforandroid.recipe import CppCompiledComponentsPythonRecipe


class DlibRecipe(CppCompiledComponentsPythonRecipe):
    version = '19.24'
    url = 'https://github.com/davisking/dlib/archive/v{version}.tar.gz'
    depends = ['numpy', 'python3']
    site_packages_name = 'dlib'
    
    def get_recipe_env(self, arch):
        env = super().get_recipe_env(arch)
        env['CMAKE_BUILD_TYPE'] = 'Release'
        return env


recipe = DlibRecipe()