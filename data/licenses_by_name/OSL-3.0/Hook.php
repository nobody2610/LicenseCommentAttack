* 2007-2017 PrestaShop
 *
 * NOTICE OF LICENSE
 *
 * This source file is subject to the Open Software License (OSL 3.0)
 * that is bundled with this package in the file LICENSE.txt.
 * It is also available through the world-wide-web at this URL:
 * http://opensource.org/licenses/osl-3.0.php
 * If you did not receive a copy of the license and are unable to
 * obtain it through the world-wide-web, please send an email
 * to license@prestashop.com so we can send you a copy immediately.
 *
 * DISCLAIMER
 *
 * Do not edit or add to this file if you wish to upgrade PrestaShop to newer
 * versions in the future. If you wish to customize PrestaShop for your
 * needs please refer to http://www.prestashop.com for more information.
 *
 *  @author		PrestaShop SA <contact@prestashop.com>
 *  @copyright  2007-2017 PrestaShop SA
 *  @license    http://opensource.org/licenses/osl-3.0.php  Open Software License (OSL 3.0)
 *  International Registered Trademark & Property of PrestaShop SA
 */

<?php
/**
class HookCore extends ObjectModel
{
    /**
     * @var string Hook name identifier
     */
    public $name;

    /**
     * @var string Hook title (displayed in BO)
     */
    public $title;

    /**
     * @var string Hook description
     */
    public $description;

    /**
     * @var bool
     */
    public $position = false;

    /**
     * @var bool Is this hook usable with live edit ?
     */
    public $live_edit = false;

    /**
     * @var array List of executed hooks on this page
     */
    public static $executed_hooks = array();

    public static $native_module;

    /**
     * @see ObjectModel::$definition
     */
    public static $definition = array(
        'table' => 'hook',
        'primary' => 'id_hook',
        'fields' => array(
            'name' =>            array('type' => self::TYPE_STRING, 'validate' => 'isHookName', 'required' => true, 'size' => 64),
            'title' =>            array('type' => self::TYPE_STRING, 'validate' => 'isGenericName'),
            'description' =>    array('type' => self::TYPE_HTML, 'validate' => 'isCleanHtml'),
            'position' =>        array('type' => self::TYPE_BOOL, 'validate' => 'isBool'),
            'live_edit' =>    array('type' => self::TYPE_BOOL, 'validate' => 'isBool'),
        ),
    );

    /**
     * @deprecated 1.5.0
     */
    protected static $_hook_modules_cache = null;

    /**
     * @deprecated 1.5.0
     */
    protected static $_hook_modules_cache_exec = null;

    public function add($autodate = true, $null_values = false)
    {
        Cache::clean('hook_idsbyname');
        return parent::add($autodate, $null_values);
    }

    /**
     * Return Hooks List
     *
     * @param bool $position
     * @return array Hooks List
     */
    public static function getHooks($position = false)
    {
        return Db::getInstance(_PS_USE_SQL_SLAVE_)->executeS('
			SELECT * FROM `'._DB_PREFIX_.'hook` h
			'.($position ? 'WHERE h.`position` = 1' : '').'
			ORDER BY `name`'
        );
    }

    /**
     * Return hook ID from name
     *
     * @param string $hook_name Hook name
     * @return int Hook ID
     */
    public static function getIdByName($hook_name)
    {
        $hook_name = strtolower($hook_name);
        if (!Validate::isHookName($hook_name)) {
            return false;
        }

        $cache_id = 'hook_idsbyname';
        if (!Cache::isStored($cache_id)) {
            // Get all hook ID by name and alias
            $hook_ids = array();
            $db = Db::getInstance();
            $result = $db->ExecuteS('
			SELECT `id_hook`, `name`
			FROM `'._DB_PREFIX_.'hook`
			UNION
			SELECT `id_hook`, ha.`alias` as name
			FROM `'._DB_PREFIX_.'hook_alias` ha
			INNER JOIN `'._DB_PREFIX_.'hook` h ON ha.name = h.name', false);
            while ($row = $db->nextRow($result)) {
                $hook_ids[strtolower($row['name'])] = $row['id_hook'];
            }
            Cache::store($cache_id, $hook_ids);
        } else {
            $hook_ids = Cache::retrieve($cache_id);
        }

        return (isset($hook_ids[$hook_name]) ? $hook_ids[$hook_name] : false);
    }

    /**
     * Return hook ID from name
     */
    public static function getNameById($hook_id)
    {
        $cache_id = 'hook_namebyid_'.$hook_id;
        if (!Cache::isStored($cache_id)) {
            $result = Db::getInstance()->getValue('
							SELECT `name`
							FROM `'._DB_PREFIX_.'hook`
							WHERE `id_hook` = '.(int)$hook_id);
            Cache::store($cache_id, $result);
            return $result;
        }
        return Cache::retrieve($cache_id);
    }

    /**
     * Return hook live edit bool from ID
     */
    public static function getLiveEditById($hook_id)
    {
        $cache_id = 'hook_live_editbyid_'.$hook_id;
        if (!Cache::isStored($cache_id)) {
            $result = Db::getInstance()->getValue('
							SELECT `live_edit`
							FROM `'._DB_PREFIX_.'hook`
							WHERE `id_hook` = '.(int)$hook_id);
            Cach